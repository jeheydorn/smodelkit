package smodelkit.learner;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import smodelkit.MLSystemsManager;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.util.Counter;
import smodelkit.util.Helper;
import smodelkit.util.Range;
import smodelkit.util.Tuple2Comp;

/**
 * A simple ensemble technique in which each model in the ensemble votes
 * on the output to predict. In the case of multiple outputs, votes are counted
 * for each output separately. 
 * @author joseph
 *
 */
public class VotingEnsemble extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	private List<SupervisedLearner> subModels;
	private Matrix labels;

	public void configure(List<Tuple2Comp<String, String>> subModelSettings)
	{
		if (subModelSettings.size() < 1)
		{
			throw new IllegalArgumentException("An ensemble cannot have 0 sub-models.");
		}
		subModels = new ArrayList<>();
		for (Tuple2Comp<String, String> settings : subModelSettings)
		{
			try
			{
				subModels.add(MLSystemsManager.createLearner(rand, settings.getFirst(), settings.getSecond()));
			} 
			catch (IOException e)
			{
				throw new RuntimeException(e);
			}
		}
	}
	
	@Override
	public void configure(JSONObject settings)
	{
		JSONArray subModelSettingsJson = (JSONArray) settings.get("subModels");
		List<Tuple2Comp<String, String>> subModelSettings = new ArrayList<>();
		for (Object subModelObj : subModelSettingsJson)
		{
			JSONObject subModelJson = (JSONObject)subModelObj;
			String name = subModelJson.get("name").toString();
			String settingsFile = subModelJson.get("settingsFile").toString();
			subModelSettings.add(new Tuple2Comp<>(name, settingsFile));
		}
		configure(subModelSettings);
	}
	
	@Override
	protected void innerTrain(final Matrix inputs, final Matrix labels)
	{
		this.labels = labels;
		List<Runnable> jobs = new ArrayList<>();
		for (final SupervisedLearner subModel : subModels)
		{
			jobs.add(new Runnable()
			{
				public void run()
				{
					subModel.train(inputs, labels);
				}
			});
		}
		Helper.processInParallel(jobs);
	}

	@Override
	protected Vector innerPredict(Vector input)
	{
		List<Vector> predictions = new ArrayList<Vector>();
		for (SupervisedLearner subModel : subModels)
		{
			predictions.add(subModel.predict(input));
		}
		
		double[] result = new double[predictions.get(0).size()];
		for (int c : new Range(result.length))
		{
			if (labels.isContinuous(c))
			{
				// Find the mean of the predictions.
				double sum = 0;
				for (Vector pred : predictions)
				{
					sum += pred.get(c);
				}
				result[c] = sum/predictions.size();
			}
			else
			{
				// Find the mode of the predictions.
				Counter<Double> counts = new Counter<>();
				for (Vector pred : predictions)
				{
					counts.increment(pred.get(c));
				}
				result[c] = counts.argmax();
			}
		}
		return new Vector(result);
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
	protected boolean canImplicitlyHandleMultipleOutputs()
	{
		return true;
	}

	@Override
	protected boolean canImplicitlyHandleUnknownOutputs()
	{
		return false;
	}
	
	@Override
	public boolean canImplicitlyHandleInstanceWeights()
	{
		return true;
	}

}
