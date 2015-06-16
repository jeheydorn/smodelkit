

float getWeight(__global float* layerWeights, int nodeNumber, int weightNumber, int numWeightsPerNode)
{
	return layerWeights[numWeightsPerNode * nodeNumber + weightNumber];
}

float calcNet(__global float* layerWeights, __global float* inputs, int numInputs, int nodeNumber,
		int numWeightsPerNode)
{
	float total = 0;
	for(int i = 0; i < numInputs; i++)
		total += inputs[i] * getWeight(layerWeights, nodeNumber, i, numWeightsPerNode);
	
	// bias weight
	total += getWeight(layerWeights, nodeNumber, numWeightsPerNode - 1, numWeightsPerNode);
	
	return total;
}

/**
 * @param networkInputs An array of instances. An array is passed instead of a single instance to
 * reduce the number of TODO
 * @param instanceRow An index into networkInputs specifying the instance to make a prediction for.
 */
__kernel void calcOutputsForLayer(__global float* networkInputs, int numInputAttributes, 
		int layerIndex, __global float* networkWeights, __global int* nodeWeightCountsPerLayer, 
		__global int* nodeCountsPerLayer, __global float* outputs) 
{

    int nodeIndex = get_global_id(0);
    
    __global float* nodeInputs;
    int numLayerInputs;
    if (layerIndex == 0)
    {
    	nodeInputs = networkInputs;// + (numInputAttributes * instanceRow);
    	numLayerInputs = numInputAttributes;
    }
    else
    {
    	// The input to this layer is the output of a previous layer. Find that layer's outputs.
    	nodeInputs = outputs;
        for (int i = 0; i < layerIndex - 1; i++)
        {
        	nodeInputs += nodeCountsPerLayer[i];
        }
        numLayerInputs = nodeCountsPerLayer[layerIndex - 1];
     }
    
    // Find the start of the weights for the current layer.
    __global float* layerWeights = networkWeights;
    for (int i = 0; i < layerIndex; i++)
    {
    	layerWeights += nodeCountsPerLayer[i] * nodeWeightCountsPerLayer[i];
    }
    int numWeightsPerNode = nodeWeightCountsPerLayer[layerIndex];
    // Find the start of the outputs for the current layer.
    __global float* layerOutputs = outputs;
    for (int i = 0; i < layerIndex; i++)
    {
    	layerOutputs += nodeCountsPerLayer[i];
    }

	float net = calcNet(layerWeights, nodeInputs, numLayerInputs, nodeIndex, numWeightsPerNode);
	layerOutputs[nodeIndex] = 1.0f/(1.0f + exp(-net));

}

__kernel void calcOutputLayerErrors(__global int* nodeCountsPerLayer, __global float* errors)
{
}

//__kernel void calcHiddenLayerErrors(__global const int* nodeCountsPerLayer, int layerIndex)
//{
//}
//
//__kernel void updateWeights(__global const int* nodeCountsPerLayer)
//{
//	
//}

