package smodelkit.learner.neuralnet;
import java.util.Random;

import smodelkit.util.Bounds;


public class SoftsignNode extends NeuralNode
{
	private static final long serialVersionUID = 1L;
	
	public SoftsignNode(Random r, int numInputs, double momentum)
	{
		super(r, numInputs, momentum);
	}
		
	@Override
	public double activation(double net)
	{
		return net / (1 + Math.abs(net));
	}
	
	@Override
	public double calcOutputNodeError(double target, double output)
	{
		return (target - output)/((Math.abs(output) + 1) * (Math.abs(output) + 1));
	}

	@Override
	public double calcHiddenNodeError(double errorFromHigherLayer, double output)
	{
		return errorFromHigherLayer / ((Math.abs(output) + 1) * (Math.abs(output) + 1));
	}
	
	@Override
	public Bounds getOutputRange()
	{
		return new Bounds(-1, 1);
	}

}
