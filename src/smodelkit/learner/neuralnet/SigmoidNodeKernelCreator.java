package smodelkit.learner.neuralnet;

import smodelkit.util.Bounds;

import com.amd.aparapi.Kernel;

public class SigmoidNodeKernelCreator
{
	Bounds nodeOutputRange;
	
	public SigmoidNodeKernelCreator()
	{
		nodeOutputRange = new Bounds(0, 1);
	}
	
	public Kernel createHiddenLayerOutputKernel(final double[] nodeInputs, final double[] layerWeights, final int numWeights, final double[] layerOutputs)
	{
		return new Kernel()
		{
			int inputsLength = nodeInputs.length;
			
			@Override
			public void run()
			{
				int gid = getGlobalId();
				
				double net = calcNet(nodeInputs, gid);
				layerOutputs[gid] = 1.0/(1.0 + exp(-net));
				
			}
			
			double calcNet(double[] inputs, int nodeNumber)
			{
				double total = 0;
				for(int i = 0; i < inputsLength; i++)
					total += inputs[i] * getWeight(nodeNumber, i);

				// bias weight
				total += getWeight(nodeNumber, numWeights - 1);
				
				return total;
			}
			
			double getWeight(int nodeNumber, int weightNumber)
			{
				return layerWeights[numWeights * nodeNumber + weightNumber];
			}
		};
	}

	public Bounds getNodeOutputRange()
	{
		return nodeOutputRange;
	}



}
