package smodelkit.learner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.json.simple.JSONObject;

import smodelkit.Matrix;
import smodelkit.QuoteParser;
import smodelkit.Vector;
import smodelkit.util.Helper;
import smodelkit.util.Range;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * A wrapper for models from the Weka toolkit to allow them to be run in SMODeLKit.
 * @author joseph
 *
 */
public class WekaWrapper extends SupervisedLearner
{
	private static final long serialVersionUID = 1L;
	private Classifier classifier;
	private Instances headerData;
	private boolean normalizePredictedWeights;
	
	/**
	 * @param classifierName A class name of a weka classifier. Example:
	 *  "weka.classifiers.rules.ZeroR"
	 * @param options Options for the classifier. This can be null. Any options which equal "$RANDOM"
	 * will be replaced with a random integer.
	 * @param normalizeOutputWeights If true, then the weights assigned
	 * to each output in innerGetOutputWeights will be normalized to sum to 1.
	 */
	private void configure(String classifierName, String[] options, boolean normalizePredictedWeights)
	{
		// Copy options so we don't mutate it.
		options = Arrays.copyOf(options, options.length);
		this.normalizePredictedWeights = normalizePredictedWeights;
		
		for (int i : new Range(options.length))
		{
			// Replace occurrences of "$RANDOM" with a random integer.
			if (options[i].equals("$RANDOM"))
			{
				options[i] = String.valueOf(rand.nextInt());
			}
			// Remove double and single qoates from the beginning and end of the option.
			if (options[i].startsWith("'") && options[i].endsWith("'") 
					|| options[i].startsWith("\"") && options[i].endsWith("\""))
			{
				options[i] = options[i].substring(1, options[i].length() - 1);
			}
		}
		try
		{
			classifier = AbstractClassifier.forName(classifierName, options);
		} catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}
	
	@Override
	public void configure(JSONObject settings)
	{
		String classifierName = settings.get("classifierName").toString().trim();
		String optionsStr = ((String)settings.get("options"));
		
		// The Weka options may have quotes, so I need to parse them here.
		QuoteParser parser = new QuoteParser(optionsStr);
		String[] options = Helper.iteratorToList(parser).toArray(new String[0]);
		boolean normalizePredictedWeights = (Boolean)settings.get("normalizePredictedWeights");
		configure(classifierName, options, normalizePredictedWeights);		
	}

	@Override
	protected void innerTrain(Matrix inputs, Matrix labels)
	{
		Matrix data = new Matrix(inputs);
		data.copyColumns(labels, 0, labels.cols());
		Instances wekaInstances = convertToWekaFormat(data);
		headerData = new Instances(wekaInstances, 0);
		data = null;
		inputs = null;
		labels = null;
		try
		{
			classifier.buildClassifier(wekaInstances);
		} catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}

	@Override
	public Vector innerPredict(Vector input)
	{
		try
		{
			Instance wekaInstance = convertToWekaFormatWithEmptyClass(input, 1.0);
			return new Vector(classifier.classifyInstance(wekaInstance));
		} catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}
	
	@Override
	public final List<double[]> innerPredictOutputWeights(Vector input)
	{
		double[] outputWeights;
		try
		{
			Instance wekaInstance = convertToWekaFormatWithEmptyClass(input, 1.0);
			outputWeights = classifier.distributionForInstance(wekaInstance);
		} catch (Exception e)
		{
			throw new RuntimeException(e);
		}
		
		if (normalizePredictedWeights)
		{
			Helper.normalize(outputWeights);
		}
		
		List<double[]> result = new ArrayList<>(1);
		result.add(outputWeights);
		return result;
	}

	/**
	 * Converts the given instances to the format I use in SMODeLKit. This assumes
	 * that there is only 1 class attribute, that it is stored as the first attribute,
	 * and that it is obtained by calling instances.classAttribute().
	 * @param instances
	 * @return An array of matrixes (datasets). The zero'th element is the inputs,
	 * and the 1st element is the labels.
	 */
	private static Instances convertToWekaFormat(Matrix data)
	{		
		ArrayList<Attribute> attributes = new ArrayList<>();
		for (int c : new Range(data.cols()))
		{
			if (data.isContinuous(c))
			{
				attributes.add(new Attribute(data.getAttrName(c)));
			}
			else
			{
				// Nominal.
				List<String> nominalValues = new Range(data.getValueCount(c)).stream().map(
						v -> data.getAttrValueName(c, v)).collect(Collectors.toList());
				attributes.add(new Attribute(data.getAttrName(c), nominalValues));
			}
		}
		
		Instances instances = new Instances(data.getRelationName(), attributes, data.rows());
		instances.setClassIndex(attributes.size() - 1);
		for (int r : new Range(data.rows()))
		{
			Instance inst = convertToWekaFormat(data.row(r), instances);
			instances.add(inst);
		}
		return instances;
	}
	
	private static Instance convertToWekaFormat(Vector row, Instances dataset)
	{			
		// I use the same format for instances as Weka, and Weka's DenseInstance class cannot mutate
		// the underlying double array, so I don't need to copy it.

		Instance instance = new DenseInstance(row.getWeight(), row.getValuesForWekaInstance());	
		instance.setDataset(dataset);
		return instance;
	}

	/**
	 * Converts the given instance my the format of my dataset to that of Weka's.
	 * The class value will be 0.
	 */
	private Instance convertToWekaFormatWithEmptyClass(Vector row, double weight)
	{			
		// Add an extra value for an empty class value.
		double[] d = new double[row.size() + 1];
		for (int i = 0; i < row.size(); i++)
			d[i] = row.get(i);
		
		Instance instance = new DenseInstance(weight, d);	
		instance.setDataset(headerData);
		return instance;
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
		return false;
	}

	@Override
	public boolean canImplicitlyHandleInstanceWeights()
	{
		return true;
	}

}
