package smodelkit.filter;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import smodelkit.MLSystemsManager;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.util.Pair;
import smodelkit.util.Range;

public class ReorderOutputs extends Filter
{
	private static final long serialVersionUID = 1L;
	private List<String> outputColumnNames;
	private List<Integer> outputColumnOrder;
	private boolean randomOrder;

	@Override
	public void initializeInternal(Matrix inputs, Matrix labels)
	{
		if (outputColumnOrder != null)
		{
			// The column order was already set by setOutputColumnOrder.
		}
		else if (randomOrder)
		{
			outputColumnOrder = new ArrayList<>();

			List<Integer> colIndexes = new Range(labels.cols()).toList();
			while(!colIndexes.isEmpty())
			{
				int index = rand.nextInt(colIndexes.size());
				outputColumnOrder.add(colIndexes.get(index));
				colIndexes.remove(index);
			}

		}
		else
		{		
			if (outputColumnNames.size() == 1 && outputColumnNames.get(0).equals("reverse"))
			{
				outputColumnNames = new ArrayList<>();
				for (int c = labels.cols() - 1; c >= 0; c--)
				{
					outputColumnNames.add(labels.getAttrName(c));
				}
			}

			if (labels.cols() != outputColumnNames.size())
				throw new IllegalArgumentException("The given output columns do not match the number of columns in the labels.");

			outputColumnOrder = outputColumnNames.stream().map(name -> 
				{
					int index = labels.getAttributeColumnIndex(name);
					if (index == -1)
						throw new IllegalArgumentException("The labels have no column named: " + name);
					return index;
				}).collect(Collectors.toList());
			
		}
	}
	
	/**
	 * Changes the configuration to the column order specified.
	 * @param order
	 */
	public void setOutputColumnOrder(List<Integer> order)
	{
		this.outputColumnOrder = order;
	}

	@Override
	protected Vector filterInputInternal(Vector before)
	{
		return before;
	}

	@Override
	protected Vector unfilterLabelInternal(Vector before)
	{
		double[] result = new double[before.size()];
		for (int c = 0; c < before.size(); c++)
		{			
			result[c] = before.get(outputColumnOrder.indexOf(c));
		}
		return new Vector(result, before.getWeight());
	}
	
	/**
	 * This is necessary for when ReorderOutputs is used with NominalToCategorical and
	 * SupervisedLearner.getOutputWeights() is needed.
	 * @param before
	 * @return
	 */
	public List<double[]> unfilterOutputWeights(List<double[]> before)
	{
		List<double[]> result = new ArrayList<>(before.size());
		for (int c = 0; c < before.size(); c++)
		{			
			result.add(before.get(outputColumnOrder.indexOf(c)));
		}
		return result;		
	}

	@Override
	protected Matrix filterInputsInternal(Matrix inputs)
	{
		return inputs;
	}

	@Override
	protected Matrix filterLabelsInternal(Matrix labels)
	{
		return MLSystemsManager.moveLabelColumns(labels, outputColumnOrder);
	}
	
	public static void testPrivateMethods()
	{
		testWithOrder(Arrays.asList("class3", "class2", "class1"));
		testWithOrder(Arrays.asList("class1", "class2", "class3"));
		testWithOrder(Arrays.asList("class2", "class1", "class3"));
	}
	
	private static void testWithOrder(List<String> colNames)
	{
		Matrix data = new Matrix();
		data.loadFromArffFile("Datasets/test/output_reorder.arff");
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		Matrix labels = inputsAndLabels.getSecond();
		
		ReorderOutputs filter = new ReorderOutputs();
		filter.outputColumnNames = colNames;
		filter.initialize(inputsAndLabels.getFirst(), labels);
				
		Matrix labelsFiltered = filter.filterAllLabels(labels);
		assertEquals(labels.rows(), labelsFiltered.rows());
		for (int r : new Range(labels.rows()))
		{
			Vector label = labels.row(r);
			Vector labelFiltered = labelsFiltered.row(r);			
			assertEquals(label, filter.unfilterLabel(labelFiltered));
		}
		
	}

	@Override
	protected Vector filterLabelInternal(Vector before)
	{
		throw new UnsupportedOperationException();
	}

	@Override
	public void configure(String[] args)
	{
		if (args.length == 0)
		{
			// Random order.
			this.outputColumnNames = null;
			this.randomOrder = true;
		}
		else
		{
			this.outputColumnNames = Arrays.asList(args);
			this.randomOrder = false;
		}

	}

}
