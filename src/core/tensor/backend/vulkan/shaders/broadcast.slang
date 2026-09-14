#include "byte_access.slangh"

#include "float_math.slangh"

static const uint LFS_LOCAL_SIZE_X = 256;
#include "half_bits.slangh"

[SpecializationConstant]
const uint kOp = 4;
[SpecializationConstant]
const uint kInputDType = 0;
[SpecializationConstant]
const uint kOutputDType = 0;

#include "integer_math.slangh"

struct Parameters
{
    uint64_t lhsAddress;
    uint64_t rhsAddress;
    uint64_t outputAddress;
    uint lhsDims[8];
    uint rhsDims[8];
    uint outputDims[8];
    uint lhsRank;
    uint rhsRank;
    uint outputRank;
    uint count;
};

uint broadcastOffset(uint linear, uint rank, uint dims[8], uint outputRank,
                     uint outputDims[8])
{
    uint sourceOffset = 0;
    uint sourceStride = 1;
    for (uint reverse = 0; reverse < outputRank; ++reverse)
    {
        const uint outputDimension = outputRank - reverse - 1;
        const uint coordinate = linear % outputDims[outputDimension];
        linear /= outputDims[outputDimension];
        if (outputDimension + rank >= outputRank)
        {
            const uint sourceDimension = outputDimension + rank - outputRank;
            sourceOffset += (dims[sourceDimension] == 1u ? 0u : coordinate) * sourceStride;
            sourceStride *= dims[sourceDimension];
        }
    }
    return sourceOffset;
}

float floatOp(float lhs, float rhs)
{
    if (kOp == 4) return lhs + rhs;
    if (kOp == 5) return lhs - rhs;
    if (kOp == 6) return lhs * rhs;
    if (kOp == 7) return lhs / rhs;
    if (kOp == 25) return rhs == 2.0f ? lhs * lhs : pow(lhs, rhs);
    if (kOp == 26) return fmod(lhs, rhs);
    if (kOp == 27) return ieeeMaximum(lhs, rhs);
    if (kOp == 28) return ieeeMinimum(lhs, rhs);
    return lhs;
}

bool predicateFloat(float lhs, float rhs)
{
    if (kOp == 29) return lhs == rhs;
    if (kOp == 30) return lhs != rhs;
    if (kOp == 31) return lhs < rhs;
    if (kOp == 32) return lhs <= rhs;
    if (kOp == 33) return lhs > rhs;
    if (kOp == 34) return lhs >= rhs;
    if (kOp == 35) return floatNonzero(lhs) && floatNonzero(rhs);
    if (kOp == 36) return floatNonzero(lhs) || floatNonzero(rhs);
    return (floatNonzero(lhs)) != (floatNonzero(rhs));
}

bool predicateInt64(int64_t lhs, int64_t rhs)
{
    if (kOp == 29) return lhs == rhs;
    if (kOp == 30) return lhs != rhs;
    if (kOp == 31) return lhs < rhs;
    if (kOp == 32) return lhs <= rhs;
    if (kOp == 33) return lhs > rhs;
    if (kOp == 34) return lhs >= rhs;
    if (kOp == 35) return lhs != 0 && rhs != 0;
    if (kOp == 36) return lhs != 0 || rhs != 0;
    return (lhs != 0) != (rhs != 0);
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    for (uint index = dispatchThreadId.x; index < parameters.count; index += stride)
    {
        const uint lhsIndex = broadcastOffset(index, parameters.lhsRank, parameters.lhsDims,
                                              parameters.outputRank, parameters.outputDims);
        const uint rhsIndex = broadcastOffset(index, parameters.rhsRank, parameters.rhsDims,
                                              parameters.outputRank, parameters.outputDims);
        if (kInputDType == 0 || kInputDType == 1)
        {
            const float lhs = kInputDType == 0
                                  ? ((Ptr<float, Access.Read>)parameters.lhsAddress)[lhsIndex]
                                  : halfBitsToFloat(uint(((Ptr<uint16_t, Access.Read>)parameters.lhsAddress)[lhsIndex]));
            const float rhs = kInputDType == 0
                                  ? ((Ptr<float, Access.Read>)parameters.rhsAddress)[rhsIndex]
                                  : halfBitsToFloat(uint(((Ptr<uint16_t, Access.Read>)parameters.rhsAddress)[rhsIndex]));
            if (kOutputDType == 5)
                storeByte(parameters.outputAddress, index, predicateFloat(lhs, rhs) ? 1u : 0u);
            else if (kOutputDType == 0)
                ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] = floatOp(lhs, rhs);
            else
                ((Ptr<uint16_t, Access.ReadWrite>)parameters.outputAddress)[index] =
                    uint16_t(floatToHalfBits(floatOp(lhs, rhs)));
            continue;
        }
        if (kInputDType == 3)
        {
            const int64_t lhs = ((Ptr<int64_t, Access.Read>)parameters.lhsAddress)[lhsIndex];
            const int64_t rhs = ((Ptr<int64_t, Access.Read>)parameters.rhsAddress)[rhsIndex];
            if (kOutputDType == 5)
                storeByte(parameters.outputAddress, index, predicateInt64(lhs, rhs) ? 1u : 0u);
            else
            {
                ((Ptr<int64_t, Access.ReadWrite>)parameters.outputAddress)[index] = int64Binary(lhs, rhs);
            }
            continue;
        }
        const uint lhs = kInputDType == 2
                             ? uint(((Ptr<int, Access.Read>)parameters.lhsAddress)[lhsIndex])
                             : loadByte(parameters.lhsAddress, lhsIndex);
        const uint rhs = kInputDType == 2
                             ? uint(((Ptr<int, Access.Read>)parameters.rhsAddress)[rhsIndex])
                             : loadByte(parameters.rhsAddress, rhsIndex);
        if (kOutputDType == 5)
            storeByte(parameters.outputAddress, index,
                      predicateInt64(int64_t(int(lhs)), int64_t(int(rhs))) ? 1u : 0u);
        else if (kInputDType == 2)
            ((Ptr<int, Access.ReadWrite>)parameters.outputAddress)[index] = intBinary(int(lhs), int(rhs));
        else
            storeByte(parameters.outputAddress, index, uint(intBinary(int(lhs), int(rhs))) & 255u);
    }
}
