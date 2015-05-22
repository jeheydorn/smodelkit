package smodelkit.learner.neuralnet;
import java.util.Random;

/**
 * 
 */
public class SoftmaxNode extends Node
{
	private static final long serialVersionUID = 1L;
	
	public SoftmaxNode(Random r, int numInputs, double momentum)
	{
		super(r, numInputs, momentum);
	}
		
	@Override
	public double activation(double net)
	{
		return net;
	}
	
	@Override
	public double calcOutputNodeError(double target, double output)
	{
		return target - output;
	}

	@Override
	public double calcHiddenNodeError(double errorFromHigherLayer, double output)
	{
		throw new UnsupportedOperationException();
	}

}
