package smodelkit.learner.neuralnet;
import java.util.Random;

import smodelkit.util.Bounds;

/**
 * Uses sigmoid activations and relative entropy for error back-propagation. This node is for binary
 * classification. For a generalization to multi-class classification, see SoftmaxNode.
 * 
 * See "Accelerated Learning in Layered Neural Networks" by Sara A. Solla, Esther Levin, and Michael Fleisher.
 * @author joseph
 *
 */
public class EntropyLogisticNode extends Node
{
	private static final long serialVersionUID = 1L;
	
	public EntropyLogisticNode(Random r, int numInputs, double momentum)
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
	
	@Override
	public Bounds getOutputRange()
	{
		return new Bounds(0, 1);
	}

}
