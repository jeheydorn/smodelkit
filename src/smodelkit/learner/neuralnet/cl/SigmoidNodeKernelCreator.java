package smodelkit.learner.neuralnet.cl;

import smodelkit.util.BoundsFloat;

import com.amd.aparapi.Kernel;

public class SigmoidNodeKernelCreator
{
	BoundsFloat nodeOutputRange;
	
	public SigmoidNodeKernelCreator()
	{
		nodeOutputRange = new BoundsFloat(0f, 1f);
	}
	
	public Kernel createOutputKernel(float[] nodeInputs, float[] layerWeights, 
			int numWeights, float[] outLayerOutputs)
	{
		int numInputs = nodeInputs.length;
		
		return new Kernel()
		{
			@Override
			public void run()
			{
				int nodeIndex = getGlobalId();
				
				float net = calcNet(nodeInputs, nodeIndex);
				outLayerOutputs[nodeIndex] = 1f/(1f + exp(-net));
				
			}
			
			float calcNet(float[] inputs, int nodeNumber)
			{
				float total = 0;
				for(int i = 0; i < numInputs; i++)
					total += inputs[i] * getWeight(nodeNumber, i);

				// bias weight
				total += getWeight(nodeNumber, numWeights - 1);
				
				return total;
			}
			
			float getWeight(int nodeNumber, int weightNumber)
			{
				return layerWeights[numWeights * nodeNumber + weightNumber];
			}
		};
	}
	
	/**
	 * Calculates errors for the output layer. 
	 * 
	 * This is implemented on the CPU because
	 * networks usually have only a few output nodes, and their error calculations take
	 * a linear amount of time with respect to the number of output nodes.
	 */
	public void calcOutputLayerErrors(float[] targets, float[] outputs, float[] outErrors)
	{
		assert targets.length == outputs.length;
		
		for (int i = 0; i < outputs.length; i++)
		{
			outErrors[i] = outputs[i] * (1 - outputs[i]) * (targets[i] - outputs[i]);
		}
	}
	
	public Kernel createHiddenLayerErrorKernel(float[] higherLayerWeights, float[] higherLayerErrors, 
			float[] outputs, float[] outErrors)
	{
		int numHigherLayerNodes = higherLayerErrors.length;
		int higherLayerWeightsPerNode = higherLayerWeights.length / higherLayerErrors.length;
		
		return new Kernel()
		{
			@Override
			public void run()
			{
				int nodeIndex = getGlobalId();
				
				outErrors[nodeIndex] = outputs[nodeIndex] * (1f - outputs[nodeIndex]) 
						* dotProductErrorFromHigherLayer(nodeIndex);
			}
			
			float dotProductErrorFromHigherLayer(int nodeNumber)
			{
				float sum = 0;
				for (int i = 0; i < numHigherLayerNodes; i++)
				{
					sum += getHigherLayerWeight(i, nodeNumber) * higherLayerErrors[i];
				}
				return sum;
			}

			float getHigherLayerWeight(int higherLayerNodeNumber, int weightNumber)
			{
				return higherLayerWeights[higherLayerWeightsPerNode * higherLayerNodeNumber + weightNumber];
			}
		};	
	}
	
	public Kernel createWeightUpdateKernel(float[] inputs, float[] errors, float[] layerWeights, float learningRate)
	{
		int numInputs = inputs.length;
		int numWeightsPerNode = numInputs + 1; // + 1 for the bias weights.
		int numErrors = errors.length;
		assert layerWeights.length == (numInputs + 1) * errors.length;
		
		return new Kernel()
		{
			@Override
			public void run()
			{
				int nodeIndex = getGlobalId();
				
				for (int i = 0; i < numInputs; i++)
				{
					layerWeights[numWeightsPerNode * nodeIndex + i] += learningRate * errors[nodeIndex] * inputs[i];
				}
				
				// bias weight
				layerWeights[numWeightsPerNode * nodeIndex + numInputs] += learningRate * errors[numErrors - 1];
			}
			
		};
	}

	public BoundsFloat getNodeOutputRange()
	{
		return nodeOutputRange;
	}



}
