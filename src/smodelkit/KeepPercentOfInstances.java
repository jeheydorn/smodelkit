package smodelkit;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Random;

import smodelkit.util.Range;

/**
 * For removing a specified percent of a dataset using the command line.
 * @author joseph
 *
 */
public class KeepPercentOfInstances
{

	public static void keepPercentAndStore(String inputFilename, String outputFilename, 
			double percentToKeep, boolean shuffle, Random rand) throws FileNotFoundException
	{
		Matrix data = new Matrix();
		data.loadFromArffFile(inputFilename);
		if (shuffle)
			data.shuffle(rand);
		
		int numInstancesToKeep = (int)(data.rows() * (percentToKeep / 100.0)); 
		
		Matrix result = new Matrix();
		result.copyMetadata(data);
		for (int i : new Range(numInstancesToKeep))
			result.addRow(data.row(i));
		data = null;
		
		try (PrintWriter w = new PrintWriter(outputFilename))
		{
			w.println(result.toString());
		}
	}
	
	public static void main(String[] args) throws FileNotFoundException
	{
		if (args.length != 4)
			throw new IllegalArgumentException("Usage: KeepPercentOfInstances inputFilename outputFilename percentToKeep shuffle?");
		double percentToKeep = Double.parseDouble(args[2]);
		if (percentToKeep < 0 || percentToKeep > 100)
			throw new IllegalArgumentException("Percent to keep must be between 0 and 100 inclusive.");
		boolean shuffle = Boolean.parseBoolean(args[3]);
		keepPercentAndStore(args[0], args[1], percentToKeep, shuffle, new Random(0));
	}

}
