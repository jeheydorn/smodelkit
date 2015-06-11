package smodelkit.learner;

import org.json.simple.JSONObject;

import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.VectorDouble;
import smodelkit.util.Counter;
import smodelkit.util.Range;

/**
 * Simple classifier that always chooses the most common class. For multiple outputs,
 * see configure(boolean) for options on how to make predictions.
 * @author joseph
 *
 */
@SuppressWarnings("serial")
public class ZeroR extends SupervisedLearner
{
	Vector prediction;
	private boolean predictMostCommonOutputVector;

	/**
	 * 
	 * @param predictMostCommonOutputVector If true, this will predict the most common whole
	 * output vector seen in the training data. If false, it will predict the most common 
	 * value seen per output dimension independently. 
	 */
	public void configure(boolean predictMostCommonOutputVector)
	{
		this.predictMostCommonOutputVector = predictMostCommonOutputVector;
	}
	
	@Override
	public void configure(JSONObject settings)
	{
		boolean predictMostCommonOutputVector = (Boolean)settings.get("predictMostCommonOutputVector");
		configure(predictMostCommonOutputVector);
	}

	@Override
	protected void innerTrain(Matrix inputs, Matrix labels)
	{
		
		if (predictMostCommonOutputVector)
		{
			Counter<Vector> counts = new Counter<>();
			labels.stream().forEach(label -> counts.increment(label));
			prediction = counts.argmax();
		}
		else
		{
			double[] pred = new double[labels.cols()];
			for (int c : new Range(labels.cols()))
			{
				if (labels.isContinuous(c))
				{
					pred[c] = labels.findMean(c);
				}
				else
				{
					pred[c] = labels.findMode(c);
				}
			}
			prediction = new VectorDouble(pred);
		}
		
	}

	@Override
	protected Vector innerPredict(Vector input)
	{
		return prediction;
	}
	
	@Override
	protected boolean canImplicitlyHandleNominalFeatures()
	{
		return true;
	}

	@Override
	protected boolean canImplicitlyHandleContinuousFeatures()
	{
		return true;
	}

	@Override
	protected boolean canImplicitlyHandleNominalLabels()
	{
		return true;
	}

	@Override
	protected boolean canImplicitlyHandleContinuousLabels()
	{
		return true;
	}

	@Override
	protected boolean canImplicitlyHandleUnknownInputs()
	{
		return true;
	}

	@Override
	protected boolean canImplicitlyHandleUnknownOutputs()
	{
		return true;
	}

	@Override
	protected boolean canImplicitlyHandleMultipleOutputs()
	{
		return true;
	}

	@Override
	public boolean canImplicitlyHandleInstanceWeights()
	{
		return false;
	}
}
