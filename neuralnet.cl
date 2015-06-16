

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

__global float* findArrayForLayer(__global float* outputs, __global int* nodeCountsPerLayer, int layerIndex)
{
    __global float* layerOutputs = outputs;
    for (int i = 0; i < layerIndex; i++)
    {
    	layerOutputs += nodeCountsPerLayer[i];
    }
    return layerOutputs;
}

__global float* findLayerWeights(__global float* networkWeights, __global int* nodeWeightCountsPerLayer, 
		__global int* nodeCountsPerLayer, int layerIndex)
{
    __global float* layerWeights = networkWeights;
    for (int i = 0; i < layerIndex; i++)
    {
    	layerWeights += nodeCountsPerLayer[i] * nodeWeightCountsPerLayer[i];
    }
    return layerWeights;
}

/**
 * @param networkInputs An input instance, stored as one float per attribute.
 * @param numInputAttributes The length of networkInputs.
 * @param layerIndex the index of the network layer to calculate output for.
 * @param networkWeights A 3D array with all weights in the network.
 * @param nodeWeightCountsPerlayer A 1D array contianing the number of weights per
 * node in each layer.
 * @param nodeCountsPerLayer A 1D array containing the number of nodes per layer.
 * @param outputs A 2D array where the outputs of each node in each layer will be stored.
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
    __global float* layerWeights = findLayerWeights(networkWeights, nodeWeightCountsPerLayer,
    		nodeCountsPerLayer, layerIndex);
    int numWeightsPerNode = nodeWeightCountsPerLayer[layerIndex];
    // Find the start of the outputs for the current layer.
    __global float* layerOutputs = findArrayForLayer(outputs, nodeCountsPerLayer, layerIndex);

	float net = calcNet(layerWeights, nodeInputs, numLayerInputs, nodeIndex, numWeightsPerNode);
	layerOutputs[nodeIndex] = 1.0f/(1.0f + exp(-net));

}

/**
 * @param layerIndex the index of the network layer to calculate output for.
 * @param networkWeights A 3D array with all weights in the network.
 * @param nodeWeightCountsPerlayer A 1D array contianing the number of weights per
 * node in each layer.
 * @param nodeCountsPerLayer A 1D array containing the number of nodes per layer.
 * @param outputs a 2D array containing the previously calculated output vaules of every node in
 * every layer.
 * @param errors A 2D array where the errors of every node in every layer will be stored.
 * @param targets A 1D array contain the target label.
 */
__kernel void calcOutputLayerErrors(int layerIndex,
		__global float* networkWeights, __global int* nodeWeightCountsPerLayer, 
		__global int* nodeCountsPerLayer, __global float* outputs, __global float* targets,
		__global float* errors)
{
    int nodeIndex = get_global_id(0);
    __global float* layerOutputs = findArrayForLayer(outputs, nodeCountsPerLayer, layerIndex);
    __global float* layerErrors = findArrayForLayer(errors, nodeCountsPerLayer, layerIndex);
    
	for (int i = 0; i < nodeCountsPerLayer[layerIndex]; i++)
	{
		layerErrors[i] = layerOutputs[i] * (1 - layerOutputs[i]) * (targets[i] - layerOutputs[i]);
	}
}

float dotProductErrorFromHigherLayer(int nodeNumber, int numHigherLayerNodes, int numHigherLayerWeightsPerNode,
		__global float* higherLayerWeights, __global float* higherLayerErrors)
{
	float sum = 0;
	for (int i = 0; i < numHigherLayerNodes; i++)
	{
		sum += getWeight(higherLayerWeights, i, nodeNumber, numHigherLayerWeightsPerNode) 
				* higherLayerErrors[i];
	}
	return sum;
}

/**
 * @param layerIndex the index of the network layer to calculate output for.
 * @param networkWeights A 3D array with all weights in the network.
 * @param nodeWeightCountsPerlayer A 1D array contianing the number of weights per
 * node in each layer.
 * @param nodeCountsPerLayer A 1D array containing the number of nodes per layer.
 * @param outputs a 2D array containing the previously calculated output vaules of every node in
 * every layer.
 * @param errors A 2D array where the errors of every node in every layer will be stored.
 */
__kernel void calcHiddenLayerErrors(int layerIndex,
		__global float* networkWeights, __global int* nodeWeightCountsPerLayer, 
		__global int* nodeCountsPerLayer, __global float* outputs,
		__global float* errors)
{
    int nodeIndex = get_global_id(0);
    
    __global float* layerOutputs = findArrayForLayer(outputs, nodeCountsPerLayer, layerIndex);
    __global float* layerErrors = findArrayForLayer(errors, nodeCountsPerLayer, layerIndex);
    __global float* higherLayerWeights = findLayerWeights(networkWeights, nodeWeightCountsPerLayer,
    		nodeCountsPerLayer, layerIndex + 1);

	errors[nodeIndex] = layerOutputs[nodeIndex] * (1.0f - layerOutputs[nodeIndex]) 
			* dotProductErrorFromHigherLayer(nodeIndex, nodeCountsPerLayer[layerIndex + 1],
					nodeWeightCountsPerLayer[layerIndex + 1], higherLayerWeights, layerErrors);
}

__kernel void updateWeights(int layerIndex, __global int* nodeWeightCountsPerLayer, 
		__global int* nodeCountsPerLayer, float learningRate)
{
    int nodeIndex = get_global_id(0);

    __global float* layerErrors = findArrayForLayer(errors, nodeCountsPerLayer, layerIndex);
    __global float* layerWeights = findLayerWeights(networkWeights, nodeWeightCountsPerLayer,
    		nodeCountsPerLayer, layerIndex);
    
    // Find the node inputs and the number of them.
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


    for (int i = 0; i < numLayerInputs; i++)
	{
		layerWeights[nodeWeightCountsPerLayer[layerIndex] * nodeIndex + i] 
				+= learningRate * layerErrors[nodeIndex] * inputs[i];
	}
	
	// bias weight
	layerWeights[nodeWeightCountsPerLayer[layerIndex] * nodeIndex + numLayerInputs] 
			+= learningRate * layerErrors[ ... ];

}

