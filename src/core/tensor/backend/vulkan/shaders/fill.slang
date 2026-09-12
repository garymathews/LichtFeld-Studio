#include "byte_access.slangh"

static const uint LFS_LOCAL_SIZE_X = 256;

[SpecializationConstant]
const uint kDType = 0;

struct FillParameters
{
    uint64_t outputAddress;
    uint64_t pattern;
    uint64_t dims[8];
    uint64_t strides[8];
    uint count;
    uint rank;
    uint strided;
    uint padding;
};

uint64_t logicalOffset(uint linear, uniform FillParameters parameters)
{
    if (parameters.strided == 0u)
        return uint64_t(linear);
    uint64_t offset = 0;
    for (uint reverse = 0; reverse < parameters.rank; ++reverse)
    {
        const uint dimension = parameters.rank - reverse - 1;
        // linear < count <= UINT_MAX, so larger dimensions still yield quotient zero.
        // Keep offsets wide without paying for emulated 64-bit division on Metal.
        const uint extent = uint(min(parameters.dims[dimension], uint64_t(0xffffffffu)));
        const uint coordinate = linear % extent;
        linear /= extent;
        offset += uint64_t(coordinate) * parameters.strides[dimension];
    }
    return offset;
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform FillParameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint64_t stride = uint64_t(numWorkgroups.x) * LFS_LOCAL_SIZE_X;
    for (uint64_t index = dispatchThreadId.x; index < parameters.count; index += stride)
    {
        const uint64_t offset = logicalOffset(uint(index), parameters);
        if (kDType == 0 || kDType == 2 || kDType == 6)
            ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[offset] = uint(parameters.pattern);
        else if (kDType == 1)
            ((Ptr<uint16_t, Access.ReadWrite>)parameters.outputAddress)[offset] = uint16_t(parameters.pattern);
        else if (kDType == 3)
            ((Ptr<uint64_t, Access.ReadWrite>)parameters.outputAddress)[offset] = parameters.pattern;
        else
            storeByte(parameters.outputAddress, offset, uint(parameters.pattern));
    }
}
