#include "compensated_sum.slangh"

static const uint LFS_LOCAL_SIZE_X = 256;

// 0 uniform [first, second), 1 bernoulli(first), 2 randint [low, high),
// 3 normal(first, second), 4 multinomial with replacement over weights,
// 5 Gumbel keys for sampling without replacement, 6 rank selection of the
// sampleCount largest keys into the int64 output, 7 weight statistics
// (output[0] = maximum weight bits, output[1] = invalid flag),
// 8 compensated CDF within 256-element blocks, 9 CDF of block totals.
[SpecializationConstant]
const uint kKind = 0;

groupshared uint sharedMaximum[LFS_LOCAL_SIZE_X];
groupshared uint sharedInvalid[LFS_LOCAL_SIZE_X];

struct Parameters
{
    uint64_t outputAddress;
    uint64_t weightsAddress;
    uint64_t keysAddress;
    uint64_t seed;
    uint count;
    uint sampleCount;
    int low;
    int high;
    float first;
    float second;
    uint scaleExponent;
    uint pad0;
};

static const float kUnitScale = 1.0f / 16777216.0f;
static const float kTwoPi = 6.28318530717958647692f;

// Philox4x32-10 (Salmon et al. 2011): the element index is the counter and
// the seed is the key, so every element draws an independent, reproducible
// 128-bit block.
void multiplyHighLow(uint lhs, uint rhs, out uint high, out uint low)
{
    const uint64_t product = uint64_t(lhs) * uint64_t(rhs);
    high = uint(product >> 32);
    low = uint(product & uint64_t(0xffffffffu));
}

uint4 philox(uint4 counter, uint2 key)
{
    for (uint round = 0; round < 10; ++round)
    {
        uint high0;
        uint low0;
        uint high1;
        uint low1;
        multiplyHighLow(0xD2511F53u, counter.x, high0, low0);
        multiplyHighLow(0xCD9E8D57u, counter.z, high1, low1);
        counter = uint4(high1 ^ counter.y ^ key.x, low1, high0 ^ counter.w ^ key.y, low0);
        key.x += 0x9E3779B9u;
        key.y += 0xBB67AE85u;
    }
    return counter;
}

uint4 draw(uint64_t index, uint64_t seed)
{
    const uint4 counter = uint4(uint(index & uint64_t(0xffffffffu)), uint(index >> 32), 0u, 0u);
    const uint2 key = uint2(uint(seed & uint64_t(0xffffffffu)), uint(seed >> 32));
    return philox(counter, key);
}

// 24-bit mantissa draw in [0, 1), the same construction as the CUDA kernels.
float unitInterval(uint word)
{
    return float(word >> 8) * kUnitScale;
}

// The float just below value, in the direction of smaller numbers.
float floatBelow(float value)
{
    const uint bits = asuint(value);
    if (value > 0.0f)
        return asfloat(bits - 1u);
    if (value == 0.0f)
        return asfloat(0x80000001u);
    return asfloat(bits + 1u);
}

// Preserve the low part through accumulation and comparisons; rounding each
// prefix to one float can invent intervals for zero weights or reverse ordering.
float2 addPair(float2 a, float2 b)
{
    float total, error;
    twoSum(a.x, b.x, total, error);
    float2 result;
    twoSum(total, opaque(a.y + b.y) + error, result.x, result.y);
    return result;
}

bool lessPair(float2 a, float2 b)
{
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}

float scaledWeight(uint bits, uint exponent)
{
    const uint mantissa = bits & 0x7fffffu;
    const uint sourceExponent = (bits >> 23) & 255u;
    // Decode subnormals as integers before scaling: Metal may flush a float load.
    return ldexp(float(sourceExponent == 0u ? mantissa : mantissa | 0x800000u),
                 int(max(sourceExponent, 1u)) - int(exponent) - 23);
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupThreadId : SV_GroupThreadID,
          uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    gOpaqueZero = parameters.pad0;
    if (kKind == 7)
    {
        Ptr<float, Access.Read> weights = (Ptr<float, Access.Read>)parameters.weightsAddress;
        const uint thread = groupThreadId.x;
        uint maximum = 0u;
        uint invalid = 0u;
        for (uint index = thread; index < parameters.count; index += LFS_LOCAL_SIZE_X)
        {
            const uint bits = asuint(weights[index]);
            const uint magnitude = bits & 0x7fffffffu;
            if (magnitude > 0x7f7fffffu || (bits > 0x80000000u))
                invalid = 1u;
            maximum = max(maximum, magnitude);
        }
        sharedMaximum[thread] = maximum;
        sharedInvalid[thread] = invalid;
        GroupMemoryBarrierWithGroupSync();
        for (uint span = LFS_LOCAL_SIZE_X / 2; span > 0; span >>= 1)
        {
            if (thread < span)
            {
                sharedMaximum[thread] = max(sharedMaximum[thread], sharedMaximum[thread + span]);
                sharedInvalid[thread] |= sharedInvalid[thread + span];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (thread == 0)
        {
            ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[0] = sharedMaximum[0];
            ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[1] = sharedInvalid[0];
        }
        return;
    }
    const uint blocks = (parameters.count - 1u) / LFS_LOCAL_SIZE_X + 1u;
    if (kKind == 8 || kKind == 9)
    {
        Ptr<float2, Access.ReadWrite> cdf = (Ptr<float2, Access.ReadWrite>)parameters.keysAddress;
        Ptr<uint, Access.Read> weights = (Ptr<uint, Access.Read>)parameters.weightsAddress;
        if (kKind == 9)
        {
            if (dispatchThreadId.x == 0u)
            {
                float2 sum = float2(0.0f);
                for (uint block = 0u; block < blocks; ++block)
                {
                    sum = addPair(sum, cdf[uint64_t(parameters.count) + block]);
                    cdf[uint64_t(parameters.count) + block] = sum;
                }
            }
            return;
        }
        for (uint block = dispatchThreadId.x; block < blocks; block += stride)
        {
            float2 sum = float2(0.0f);
            const uint64_t end = min(uint64_t(parameters.count), uint64_t(block + 1u) * LFS_LOCAL_SIZE_X);
            for (uint64_t i = uint64_t(block) * LFS_LOCAL_SIZE_X; i < end; ++i)
            {
                const uint bits = weights[i] & 0x7fffffffu;
                if (bits != 0u)
                    sum = addPair(sum, float2(scaledWeight(bits, parameters.scaleExponent), 0.0f));
                cdf[i] = sum;
            }
            cdf[uint64_t(parameters.count) + block] = sum;
        }
        return;
    }
    const uint total = kKind == 4 ? parameters.sampleCount : parameters.count;
    for (uint index = dispatchThreadId.x; index < total; index += stride)
    {
        if (kKind == 6)
        {
            Ptr<float, Access.Read> keys = (Ptr<float, Access.Read>)parameters.keysAddress;
            const float key = keys[index];
            uint rank = 0;
            for (uint other = 0; other < parameters.count; ++other)
            {
                const float candidate = keys[other];
                if (candidate > key || (candidate == key && other < index))
                    ++rank;
            }
            if (rank < parameters.sampleCount)
                ((Ptr<int64_t, Access.ReadWrite>)parameters.outputAddress)[rank] = int64_t(index);
            continue;
        }
        const uint4 words = draw(index, parameters.seed);
        if (kKind == 0)
        {
            float value = parameters.first;
            if (parameters.first != parameters.second)
            {
                value = fma(unitInterval(words.x), parameters.second - parameters.first, parameters.first);
                if (!(value < parameters.second))
                    value = floatBelow(parameters.second);
            }
            ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] = value;
        }
        else if (kKind == 1)
        {
            ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] =
                unitInterval(words.x) < parameters.first ? 1.0f : 0.0f;
        }
        else if (kKind == 2)
        {
            const uint64_t range = uint64_t(int64_t(parameters.high) - int64_t(parameters.low));
            const uint64_t offset = (uint64_t(words.x) * range) >> 32;
            ((Ptr<int, Access.ReadWrite>)parameters.outputAddress)[index] =
                int(int64_t(parameters.low) + int64_t(offset));
        }
        else if (kKind == 3)
        {
            const float u1 = float((words.x >> 8) + 1u) * kUnitScale;
            const float u2 = unitInterval(words.y);
            const float radius = sqrt(-2.0f * log(u1));
            const float z = radius * cos(kTwoPi * u2);
            ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] =
                parameters.first + parameters.second * z;
        }
        else if (kKind == 4)
        {
            Ptr<float2, Access.Read> cdf = (Ptr<float2, Access.Read>)parameters.keysAddress;
            const float2 total = cdf[uint64_t(parameters.count) + blocks - 1u];
            const float unit = unitInterval(words.x);
            const float high = opaque(unit * total.x);
            const float2 target = addPair(float2(high, fma(unit, total.x, -high)), float2(unit * total.y, 0.0f));
            uint low = 0u, highBlock = blocks - 1u;
            while (low < highBlock)
            {
                const uint middle = low + (highBlock - low) / 2u;
                if (lessPair(target, cdf[uint64_t(parameters.count) + middle]))
                    highBlock = middle;
                else
                    low = middle + 1u;
            }
            const float2 base = low == 0u ? float2(0.0f) : cdf[uint64_t(parameters.count) + low - 1u];
            uint highIndex = uint(min(uint64_t(parameters.count), uint64_t(low + 1u) * LFS_LOCAL_SIZE_X)) - 1u;
            low *= LFS_LOCAL_SIZE_X;
            while (low < highIndex)
            {
                const uint middle = low + (highIndex - low) / 2u;
                if (lessPair(target, addPair(base, cdf[middle])))
                    highIndex = middle;
                else
                    low = middle + 1u;
            }
            ((Ptr<int64_t, Access.ReadWrite>)parameters.outputAddress)[index] = int64_t(low);
        }
        else
        {
            Ptr<float, Access.Read> weights = (Ptr<float, Access.Read>)parameters.weightsAddress;
            float u = unitInterval(words.x);
            u = max(u, 1e-10f);
            u = min(u, 1.0f - 1e-10f);
            const float gumbel = -log(-log(u));
            const float logWeight = log(max(weights[index], 1e-10f));
            ((Ptr<float, Access.ReadWrite>)parameters.keysAddress)[index] = logWeight + gumbel;
        }
    }
}
