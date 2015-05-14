package smodelkit;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;

import smodelkit.util.Helper;
import smodelkit.util.Logger;
import smodelkit.util.Range;
import au.com.bytecode.opencsv.CSVWriter;

/**
 * Prints meta-data for a dataset.
 * @author joseph
 *
 */
public class MetadataPrinter
{

	public static void printMetadata(Matrix data)
	{
		Logger.println("Instances: " + data.rows());
		Logger.println("Attributes: " + data.cols());
		Logger.println("Inputs: " + (data.cols() - data.getNumLabelColumns()));
		Logger.println("Outputs: " + data.getNumLabelColumns());
		
		if (data.getNumLabelColumns() > 1)
		{
			int firstOutputIndex = data.cols() - data.getNumLabelColumns();
			double averageOutputValues = new Range(data.getNumLabelColumns()).stream().mapToDouble(
					c -> data.getValueCount(firstOutputIndex + c)).average().getAsDouble();
			Logger.println("Average output values: " + averageOutputValues);
			int mostOutputValues = new Range(data.getNumLabelColumns()).stream().mapToInt(
					c -> data.getValueCount(firstOutputIndex + c)).max().getAsInt();
			Logger.println("Most output values: " + mostOutputValues);
			int leastOutputValues = new Range(data.getNumLabelColumns()).stream().mapToInt(
					c -> data.getValueCount(firstOutputIndex + c)).min().getAsInt();
			Logger.println("Least output values: " + leastOutputValues);
		}
		else
		{
			Logger.println("Number of output values: " + data.getValueCount(data.cols() - 1));
		}
		
		Logger.println();
	}
	
	public static void printMetadataPerAttribute(Matrix data)
	{
		// Print info on each column.
		for (int c = 0; c < data.cols(); c++)
		{
			// Find length of longest attribute name.
			int longestLength = 0;
			for (int value = 0; value < data.getValueCount(c); value++)
			{
				if (data.getAttrValueName(c,value).length() > longestLength)
				{
					longestLength = data.getAttrValueName(c,value).length();
				}
			}
			int largestCountLength = Integer.toString(data.countValues(c, data.findMode(c))).length();
			if (!data.isContinuous(c))
			{
				Logger.println("Distribution of column " +  data.getAttrName(c) + ":");
				for (int value = 0; value < data.getValueCount(c); value++)
				{
					int count = data.countValues(c, value);
					Logger.println(data.getAttrValueName(c,value) + ": " 
					+ Helper.getNChars(longestLength + largestCountLength - (data.getAttrValueName(c,value).length() + Integer.toString(count).length()), ' ') +
						+ count + " (" 
						+ Helper.formatDouble(((double)count)/data.rows()) + ")");
				}
				Logger.println();
			}
		}

	}
	
	public static void printExtraInfoForNumericTargets(Matrix data)
	{
		Matrix discretized = DiscretizeNumericDataset.discretizeNumericLabels(data);
		for (int c : new Range(data.cols() - data.getNumLabelColumns(), data.cols()))
		{
			if (data.isContinuous(c))
			{
				Logger.println("Unique value count for output " + data.getAttrName(c) 
						+ ": " + discretized.getValueCount(c));
			}
		}
	}
	
	public static void createTableForDatasets(List<Path> paths)
	{
		// For each dataset, show: name, #inputs, #outputs, #output values (a range in some cases), # unique output vectors
		try (CSVWriter writer = new CSVWriter(new FileWriter("table.csv"), ','))
		{
			writer.writeNext(new String[] {"Name", "Instances", "m", "d", "Values per output", "Unique output vectors"});
			
			for (Path path  : paths)
			{
				Matrix data = new Matrix();
				data.loadFromArffFile(path.toString());
				int firstOutputIndex = data.cols() - data.getNumLabelColumns();
				int mostOutputValues = new Range(data.getNumLabelColumns()).stream().mapToInt(
						c -> data.getValueCount(firstOutputIndex + c)).max().getAsInt();
				int leastOutputValues = new Range(data.getNumLabelColumns()).stream().mapToInt(
						c -> data.getValueCount(firstOutputIndex + c)).min().getAsInt();		
				
				String[] line = new String[]
				{
					path.getFileName().toString().replace(".arff", ""),
					String.valueOf(data.rows()),
					String.valueOf((data.cols() - data.getNumLabelColumns())),
					String.valueOf(data.getNumLabelColumns()),
					leastOutputValues == mostOutputValues ? String.valueOf(leastOutputValues) : leastOutputValues + "-" + mostOutputValues,
					String.valueOf(countUniqueOutputVectors(data))
				};
				writer.writeNext(line);
			}
		} 
		catch (IOException e)
		{
			throw new RuntimeException(e);
		}
	}
	
	private static int countUniqueOutputVectors(Matrix data)
	{
		Matrix labels = data.splitInputsAndLabels().getSecond();
		Set<Vector> unique = new TreeSet<>();
		labels.stream().forEach(row -> unique.add(row));
		return unique.size();
	}
	
	public static void main(String[] args) throws IOException
	{
		// Add MDC datasets.
		List<Path> paths = Files.list(Paths.get("Datasets/mdc/")).filter(
				path -> !Files.isDirectory(path)).collect(Collectors.toList());
		// Add MLC datasets.
//		Files.list(Paths.get("Datasets/mlc")).filter(
//				path -> !Files.isDirectory(path)).forEach(path -> paths.add(path));
		
		// Sort by filename alphabetically.
		paths.sort((path1, path2) -> path1.getFileName().compareTo(path2.getFileName()));
		
		createTableForDatasets(paths);
		
//		for (Path datasetPath : paths)
//		{
//			Logger.println("File: " + datasetPath.getFileName());
//			Matrix data = new Matrix();
//			data.loadFromArffFile(datasetPath.toString());
//			printMetadata(data);
//		}
		
		Logger.println("Done.");
		
	}

}
