package smodelkit.learner.neuralnet;

import java.util.Random;

@SuppressWarnings("serial")
public class LinearErrorSigmoidNode extends Node
{

	public LinearErrorSigmoidNode(Random r, int numInputs, double momentum)
	{
		super(r, numInputs, momentum);
	}

	@Override
	public double squash(double net)
	{
		return 1/(1 + Math.exp(-net));
	}

	@Override
	public double calcOutputNodeError(double target, double output)
	{
		if (target >= output)
		{
			return output * (1 - output);
		}
		else
		{
			return -output * (1 - output);
		}
	}

	@Override
	public double calcHiddenNodeError(double errorFromHigherLayer, double output)
	{
		return output * (1 - output) *  errorFromHigherLayer;
	}

}
