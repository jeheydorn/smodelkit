package smodelkit;

import java.util.Random;

import smodelkit.util.Range;

/**
 * For sampling from a dataset. 
 * @author joseph
 *
 */
public class Sample
{
	/**
	 * Samples a given dataset with replacement. The size of the result is specified by a
	 * percent of the size of the original dataset.
	 * @param inputs
	 * @param labels
	 * @param percent
	 * @return The first element is the resulting inputs. The second element is the resulting labels.
	 */
	public static Matrix[] sampleWithReplacement(Random rand, Matrix inputs, Matrix labels, double percent)
	{
		if (inputs.rows() != labels.rows())
		{
			throw new IllegalArgumentException();
		}
		
		Matrix[] result = new Matrix[2];
		result[0] = new Matrix();
		result[0].copyMetadata(inputs);
		result[1] = new Matrix();
		result[1].copyMetadata(labels);
		
		int resultRows = (int)Math.round(inputs.rows() * percent);
		for (@SuppressWarnings("unused") int i : new Range(resultRows))
		{
			int r = rand.nextInt(inputs.rows());
			result[0].addRow(inputs.row(r));
			result[1].addRow(labels.row(r));
		}
		
		return result;
	}
	
	/**
	 * Like sampleWithReplacement except instead of duplicating instances in the results,
	 * instance weights are used instead of creating duplicates in the results. The percent
	 * is fixed at 100%.
	 * @param rand
	 * @param inputs
	 * @param labels
	 * @return The first element is the resulting inputs. The second element is the resulting labels.
	 */
	public static Matrix[] sampleWithReplacementUsingInstanceWeights(Random rand, Matrix inputs, Matrix labels)
	{
		if (inputs.rows() != labels.rows())
		{
			throw new IllegalArgumentException();
		}
		
		int[] weights = new int[inputs.rows()];
		for (@SuppressWarnings("unused") int i : new Range(weights.length))
		{
			weights[rand.nextInt(weights.length)]++;
		}
		
		Matrix baggedInputs = new Matrix();
		Matrix baggedLabels = new Matrix();
		baggedInputs.copyMetadata(inputs);
		baggedLabels.copyMetadata(labels);
		for (int i : new Range(weights.length))
		{
			if (weights[i] > 0)
			{
				Vector x = new Vector(inputs.row(i));
				x.setWeight(weights[i]);
				baggedInputs.addRow(x);
				Vector y = new Vector(labels.row(i));
				y.setWeight(weights[i]);
				baggedLabels.addRow(y);
			}
		}
		
		return new Matrix[] {baggedInputs, baggedLabels};
	}

}
