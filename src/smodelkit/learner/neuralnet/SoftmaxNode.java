package smodelkit.learner.neuralnet;
import java.util.Random;

import smodelkit.util.Bounds;

/**
 * For performing softmax regression in output layer nodes. The error being minimized is 
 * relative entropy. 
 * 
 * This is based on http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression.
 * 
 * This node type requires additional logic in NeuralNet, so the error and activation functions here
 * are not enough to tell what the actual error and activation functions are.
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

	@Override
	public Bounds getOutputRange()
	{
		return new Bounds(0, 1);
	}
	

}
