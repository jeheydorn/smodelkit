package smodelkit.learner;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Function;

import org.json.simple.JSONObject;

import smodelkit.MLSystemsManager;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.util.Helper;
import smodelkit.util.Range;

/** 
 * This is a problem transformation technique for MDC (multi-dimensional classification). 
 * It creates a unique label (output) for every unique output vector, and then trains
 * a standard multi-class classifier on the result.
 * @author joseph
 *
 */
public class MonolithicTransformation extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	private String subModelName;
	private JSONObject subModelSettings;
	/**
	 * After transformLabels() is called, the index of a label value is it's value in the transformed labels.
	 */
	private List<Vector> uniqueLabelsList;
	private SupervisedLearner subModel;

	public void configure(String subModelName, JSONObject subModelSettings)
	{
		this.subModelName = subModelName;
		this.subModelSettings = subModelSettings;
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
		Matrix tranformedLabels = transformLabels(labels);
		labels = null;
		subModel = MLSystemsManager.createLearner(rand, subModelName, subModelSettings);
		subModel.train(inputs, tranformedLabels);
	}
	
	private Matrix transformLabels(Matrix labels)
	{
		{
			Set<Vector> uniqueLabels = new TreeSet<>();
			labels.forEach(label -> uniqueLabels.add(label));
			uniqueLabelsList = new ArrayList<>(uniqueLabels);
		}
		Matrix result = new Matrix();
		result.setRelationName("transormed_labels");
		result.addEmptyColumn("monolithic_labels");
		
		Function<Vector, String> getLabelName = label -> labels.rowToString(label).replaceAll("\\s", "");
		
		for (Vector label : uniqueLabelsList)
		{
			String valueName = getLabelName.apply(label);
			result.addAttributeValue(0, valueName);
		}
		
		for (int r : new Range(labels.rows()))
		{
			result.addRow(new Vector(new double[] {result.getAttrValueIndex(0, getLabelName.apply(labels.row(r)))}));
		}
		
		return result;
	}

	@Override
	public Vector innerPredict(Vector input)
	{
		int index = (int) subModel.predict(input).get(0);
		return uniqueLabelsList.get(index);
	}
	
	@Override
	protected List<Vector> innerPredictScoredList(Vector input, int maxDesiredSize)
	{	
		List<double[]> weightList = subModel.predictOutputWeights(input);
		assert weightList.size() == 1;
		double[] weights = weightList.get(0);
		Integer[] indexes = Helper.sortIndexesDescending(weights);
		int resultSize = Math.min(indexes.length, maxDesiredSize);
		List<Vector> result = new ArrayList<>(resultSize);
		for (int i : new Range(Math.min(indexes.length, maxDesiredSize)))
		{
			result.add(new Vector(uniqueLabelsList.get(indexes[i]), weights[indexes[i]]));
		}
		
		return result;
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
		return false;
	}

	@Override
	protected boolean canImplicitlyHandleUnknownInputs()
	{
		return true;
	}

	@Override
	protected boolean canImplicitlyHandleUnknownOutputs()
	{
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
