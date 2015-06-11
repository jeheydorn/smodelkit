package smodelkit;

import static java.lang.System.out;

import java.io.FileNotFoundException;
import java.io.PrintWriter;

import org.apache.commons.io.FilenameUtils;

import smodelkit.util.Range;

public class DiscretizeNumericDataset
{
	/**
	 * Given a multi-variate regression dataset, this converts it to a multi-dimensional
	 * classification dataset. It does so without binning, meaning it just makes a unique
	 * nominal value for every unique numeric value.
	 * @throws FileNotFoundException 
	 */
	public static void convertUniqueNumericToUniqueNominal(String arffName) throws FileNotFoundException
	{
		Matrix old = new Matrix();
		old.loadFromArffFile(arffName);
		
		Matrix result = discretizeNumericLabels(old);
		
		String arffBaseName = FilenameUtils.getBaseName(arffName);
		String outputFilename = FilenameUtils.getPath(arffName) + arffBaseName + "_nominal.arff";
		
		try (PrintWriter w = new PrintWriter(outputFilename))
		{
			w.println(result.toString());
		}

		// Make sure I can read the data back.
		Matrix tmp = new Matrix();
		tmp.loadFromArffFile(outputFilename);
		
		out.println("Result written to " + outputFilename);
		
		
	}
		
	public static Matrix discretizeNumericLabels(Matrix old)
	{
		Matrix result = new Matrix();
		result.setRelationName(old.getRelationName());
		result.setNumLabelColumns(old.getNumLabelColumns());
		result.copyColumns(old, 0, old.cols() - old.getNumLabelColumns());
		
		Matrix labels = new Matrix();
		for (int c : new Range(old.getNumLabelColumns()))
		{
			labels.addEmptyColumn(old.getAttrName(c + (old.cols() - old.getNumLabelColumns())));
			for (int r : new Range(old.rows()))
			{
				labels.addAttributeValueIfItDoesNotExist(c, numericToNominal(old.row(r).get(
						c + (old.cols() - old.getNumLabelColumns()))));
			}
		}
		
		for (int r : new Range(old.rows()))
		{
			Vector rowNumeric = old.row(r).subVector(old.cols() - old.getNumLabelColumns(), old.cols());
			// Convert rowNumeric to nominal values.
			double[] rowNominal = new double[rowNumeric.size()];
			for (int i : new Range(rowNumeric.size()))
			{
				rowNominal[i] = labels.getAttrValueIndex(i, numericToNominal(rowNumeric.get(i)));
			}
			labels.addRow(new VectorDouble(rowNominal, rowNumeric.getWeight()));
		}
		
		result.copyColumns(labels,0, labels.cols());

		return result;
	}

	
	/**
	 * Converts a numeric value to a string representing a nominal value.
	 */
	private static String numericToNominal(double numeric)
	{
		if ((int)numeric == numeric)
		{
			return (int)numeric + "";
		}
		else
		{
			return numeric + "";
		}
	}
	
	public static void main(String[] args) throws FileNotFoundException
	{
		convertUniqueNumericToUniqueNominal("Datasets/mtr/water-quality.arff");
		out.println("Done.");
	}

}
