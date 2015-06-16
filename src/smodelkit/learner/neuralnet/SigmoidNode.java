package smodelkit.learner.neuralnet;
import java.util.Random;

import smodelkit.Vector;
import smodelkit.util.Bounds;


public class SigmoidNode extends NeuralNode
{
	private static final long serialVersionUID = 1L;
	
	public SigmoidNode(Random r, int numInputs, double momentum)
	{
		super(r, numInputs, momentum);
	}
		
	@Override
	public double activation(double net)
	{
		return 1/(1 + Math.exp(-net));
	}

	@Override
	public double calcOutputNodeError(double target, double output)
	{
		return output * (1 - output) * (target - output);
	}

	@Override
	public double calcHiddenNodeError(double errorFromHigherLayer, double output)
	{
		return output * (1 - output) *  errorFromHigherLayer;
	}

	@Override
	public Bounds getOutputRange()
	{
		return new Bounds(0, 1);
	}
	
}
