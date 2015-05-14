package smodelkit;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Moves class attributes from the front to the back of of the list of attributes in an arff file.
 * @author joseph
 *
 */
public class MoveClassesToEndOfAttributes
{

	public static void moveClassesToEndOfAttributes(String inputFilename, String outputFilename)
	{
		Matrix data = new Matrix();
		data.loadFromArffFile(inputFilename, true);
		
		Matrix result = new Matrix();
		result.setRelationName(data.getRelationName());
		result.setComments(data.getComments());
		result.setNumLabelColumns(data.getNumLabelColumns());
		result.copyColumns(data, data.getNumLabelColumns(), data.cols() - data.getNumLabelColumns());
		result.copyColumns(data, 0, data.getNumLabelColumns());
				
		try (PrintWriter w = new PrintWriter(outputFilename))
		{
			w.write(result.toString());
		}
		catch (FileNotFoundException e)
		{
			throw new RuntimeException(e);
		}
	}

	public static void main(String[] args) throws IOException
	{
		if (args.length != 2)
			throw new IllegalArgumentException("usage: MoveClassesToEndOfAttributes inputArffName outputArffName.");		
		
		String arffName = args[0];
		String outputArffName = args[1];
		
		moveClassesToEndOfAttributes(arffName, outputArffName);
		
	}
}
