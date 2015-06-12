package smodelkit;

import static java.lang.System.out;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Moves class attributes from the back to the front of a list of attributes of an arff file.
 * @author joseph
 *
 */
public class MoveClassesToBeginningOfAttributes
{

	public static void moveClassesToFrontOfArff(String inputFilename, String outputFilename)
	{
		Matrix data = new Matrix();
		data.loadFromArffFile(inputFilename, true, Integer.MAX_VALUE);
		out.println("#columns: before: " + data.cols());
		
		Matrix result = new Matrix();
		result.setRelationName(data.getRelationName());
		result.setComments(data.getComments());
		result.setNumLabelColumns(data.getNumLabelColumns());
		result.copyColumns(data, data.cols() - data.getNumLabelColumns(), data.getNumLabelColumns());
		result.copyColumns(data, 0, data.cols() - data.getNumLabelColumns());
				
		out.println("#columns after: " + data.cols());

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
		Files.createDirectories(Paths.get("out"));
		
//		Files.list(Paths.get("Datasets/mlc")).forEach(
//				path -> moveClassesToBackOfArff(path.toString(), 
//						Paths.get("out", path.getFileName().toString()).toString()));

		Path path = Paths.get("Datasets/mdc/thyroid.arff");
		moveClassesToFrontOfArff(path.toString(), 
				Paths.get("out", path.getFileName().toString()).toString());

	}
}
