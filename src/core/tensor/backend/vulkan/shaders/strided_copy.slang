#include "byte_access.slangh"

static const uint LFS_LOCAL_SIZE_X = 256;

[SpecializationConstant]
const uint kInputDType = 0;
[SpecializationConstant]
const uint kOutputDType = 0;
[SpecializationConstant]
const uint kScatter = 0;

struct Parameters
{
    uint64_t inputAddress;
    uint64_t outputAddress;
    uint64_t dims[8];
    uint64_t strides[8];
    uint rank;
    uint count;
};

uint64_t logicalOffset(uint linear, uniform Parameters parameters)
{
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
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint64_t stride = uint64_t(numWorkgroups.x) * LFS_LOCAL_SIZE_X;
    for (uint64_t index = dispatchThreadId.x; index < parameters.count; index += stride)
    {
        const uint64_t strided = logicalOffset(uint(index), parameters);
        const uint64_t inputIndex = kScatter != 0 ? index : strided;
        const uint64_t outputIndex = kScatter != 0 ? strided : index;
        if (kInputDType == 2 && kOutputDType == 0)
        {
            ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[outputIndex] =
                float(((Ptr<int, Access.Read>)parameters.inputAddress)[inputIndex]);
            continue;
        }
        if (kInputDType == 0)
            ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[outputIndex] =
                ((Ptr<uint, Access.Read>)parameters.inputAddress)[inputIndex];
        else if (kInputDType == 1)
            ((Ptr<uint16_t, Access.ReadWrite>)parameters.outputAddress)[outputIndex] =
                ((Ptr<uint16_t, Access.Read>)parameters.inputAddress)[inputIndex];
        else if (kInputDType == 2)
            ((Ptr<int, Access.ReadWrite>)parameters.outputAddress)[outputIndex] =
                ((Ptr<int, Access.Read>)parameters.inputAddress)[inputIndex];
        else if (kInputDType == 3)
            ((Ptr<int64_t, Access.ReadWrite>)parameters.outputAddress)[outputIndex] =
                ((Ptr<int64_t, Access.Read>)parameters.inputAddress)[inputIndex];
        else if (kInputDType == 6)
            ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[outputIndex] =
                ((Ptr<uint, Access.Read>)parameters.inputAddress)[inputIndex];
        else
            storeByte(parameters.outputAddress, outputIndex,
                      loadByte(parameters.inputAddress, inputIndex));
    }
}
