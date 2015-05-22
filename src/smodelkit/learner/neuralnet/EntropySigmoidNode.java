package smodelkit.learner.neuralnet;
import java.util.Random;

/**
 * Uses sigmoid activations and relative entropy for error back-propagation.
 * 
 * See "Accelerated Learning in Layered Neural Networks" by Sara A. Solla, Esther Levin, and Michael Fleisher.
 * @author joseph
 *
 */
public class EntropySigmoidNode extends Node
{
	private static final long serialVersionUID = 1L;
	
	public EntropySigmoidNode(Random r, int numInputs, double momentum)
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
		// equation 2.22 in "Accelerated Learning in Layered Neural Networks".
		return target - output;
	}

	@Override
	public double calcHiddenNodeError(double errorFromHigherLayer, double output)
	{
		// equation 2.23 in "Accelerated Learning in Layered Neural Networks".
		return output * (1 - output) *  errorFromHigherLayer;
	}

}
