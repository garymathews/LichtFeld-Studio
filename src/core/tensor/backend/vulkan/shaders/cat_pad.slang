#include "byte_access.slangh"

static const uint LFS_LOCAL_SIZE_X = 256;

[SpecializationConstant]
const uint kDType = 0;
[SpecializationConstant]
const uint kMode = 0;

struct Parameters
{
    uint64_t inputAddress;
    uint64_t outputAddress;
    uint inputDims[8];
    uint inputStrides[8];
    uint outputStrides[8];
    uint padBefore[8];
    uint count;
    uint rank;
    uint inputBlock;
    uint outputBlock;
    uint outputOffset;
    uint padding;
};

void copyElement(uint64_t inputAddress, uint64_t inputIndex,
                 uint64_t outputAddress, uint64_t outputIndex)
{
    if (kDType == 0 || kDType == 2 || kDType == 6)
        ((Ptr<uint, Access.ReadWrite>)outputAddress)[outputIndex] =
            ((Ptr<uint, Access.Read>)inputAddress)[inputIndex];
    else if (kDType == 1)
        ((Ptr<uint16_t, Access.ReadWrite>)outputAddress)[outputIndex] =
            ((Ptr<uint16_t, Access.Read>)inputAddress)[inputIndex];
    else if (kDType == 3)
        ((Ptr<uint64_t, Access.ReadWrite>)outputAddress)[outputIndex] =
            ((Ptr<uint64_t, Access.Read>)inputAddress)[inputIndex];
    else
        storeByte(outputAddress, outputIndex, loadByte(inputAddress, inputIndex));
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    for (uint index = dispatchThreadId.x; index < parameters.count; index += stride)
    {
        if (kMode == 0)
        {
            const uint outer = index / parameters.inputBlock;
            const uint within = index % parameters.inputBlock;
            copyElement(parameters.inputAddress, index, parameters.outputAddress,
                        uint64_t(outer * parameters.outputBlock + parameters.outputOffset + within));
            continue;
        }
        uint remaining = index;
        uint64_t inputOffset = 0;
        uint64_t outputOffset = 0;
        for (uint dimension = 0; dimension < parameters.rank; ++dimension)
        {
            uint64_t contiguousStride = 1;
            for (uint next = dimension + 1; next < parameters.rank; ++next)
                contiguousStride *= parameters.inputDims[next];
            const uint64_t coordinate = uint64_t(remaining) / contiguousStride;
            remaining = uint(uint64_t(remaining) % contiguousStride);
            inputOffset += coordinate * parameters.inputStrides[dimension];
            outputOffset += (coordinate + parameters.padBefore[dimension]) *
                            parameters.outputStrides[dimension];
        }
        copyElement(parameters.inputAddress, inputOffset,
                    parameters.outputAddress, outputOffset);
    }
}
