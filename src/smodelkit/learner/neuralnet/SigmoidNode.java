package smodelkit.learner.neuralnet;
import java.util.Random;


public class SigmoidNode extends Node
{
	private static final long serialVersionUID = 1L;
	
	public SigmoidNode(Random r, int numInputs, double momentum)
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
		return output * (1 - output) * (target - output);
	}

	@Override
	public double calcHiddenNodeError(double errorFromHigherLayer, double output)
	{
		return output * (1 - output) *  errorFromHigherLayer;
	}

}
