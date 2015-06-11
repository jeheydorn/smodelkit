package smodelkit.learner;

import org.json.simple.JSONObject;

import smodelkit.Vector;
import smodelkit.MLSystemsManager;
import smodelkit.Matrix;
import smodelkit.Vector;

/**
 * A wrapper for other learners which detects if the training labels are all
 * the same. If so, It uses a ZeroR as a sub-model. If not, it uses a specified
 * learner. This is useful for wrapper learners which cannot handle training
 * labels being all one class.
 * @author joseph
 *
 */
public class OneClassWrapper extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	private SupervisedLearner subModel;
	
	private void configure(String subModelName, JSONObject submodelSettings) 
	{
		this.subModel = MLSystemsManager.createLearner(rand, subModelName, submodelSettings);
	}

	@Override
	public void configure(JSONObject settings)
	{
		String submodelName = (String)settings.get("submodelName");
		String submodelSettingsFile = settings.get("submodelSettingsFile").toString();
		JSONObject submodelSettings = MLSystemsManager.parseModelSettingsFile(submodelSettingsFile);
		configure(submodelName, submodelSettings);
	}

	@Override
	protected void innerTrain(Matrix inputs, Matrix labels)
	{
		if (labels.rows() == 0)
		{
			ZeroR zeroR = new ZeroR();
			zeroR.configure(false);
			subModel = zeroR;
		}
		else
		{
			// Check if all rows are the same in labels.
			Vector firstLabel = labels.row(0);
			boolean moreThanOneClass = labels.stream().anyMatch(row -> !row.equals(firstLabel));
			if (!moreThanOneClass)
			{
				ZeroR zeroR = new ZeroR();
				zeroR.configure(false);
				subModel = zeroR;
			}
		}
		subModel.train(inputs, labels);
	}

	@Override
	protected Vector innerPredict(Vector input)
	{
		return subModel.innerPredict(input);
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
		// I would need to change innerTrain to handle unknown outputs.
		return false;
	}

	@Override
	protected boolean canImplicitlyHandleMultipleOutputs()
	{
		return true;
	}

	@Override
	public boolean canImplicitlyHandleInstanceWeights()
	{
		return true;
	}

}
