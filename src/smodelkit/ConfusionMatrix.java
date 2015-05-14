package smodelkit;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import smodelkit.util.Helper;

/**
 * 
 * @author joseph
 *
 * @param <String> The type of label being predicted by the model generating information to fill this
 * 	confusion matrix.
 */
public class ConfusionMatrix
{

	private MatrixCounter matrixCounter;
	private String labelName;
		
	public ConfusionMatrix(String labelName)
	{
		matrixCounter = new MatrixCounter();
		this.labelName = labelName;
	}
	
	public void incrementCount(String refName, String hypName)
	{
		matrixCounter.incrementCount(refName, hypName);
	}
	
	public int getCount(String refName, String hypName)
	{
		return matrixCounter.getCount(refName, hypName);
	}
	
	public Map<String, Double> getAccuracyPerAttribute()
	{
		Map<String, Double> result = new TreeMap<String, Double>();
		
		for (String ref : matrixCounter.getAllAttributeNames())
		{
			result.put(ref, (double)matrixCounter.getCount(ref, ref)/matrixCounter.getTotalCount(ref));
		}
		
		return result;
	}
	
	public String getAccuracyPerAttributePrinted()
	{
		Map<String, Double> pCorrect = getAccuracyPerAttribute();
		String result = "";
		
		int longest = 0;
		for (String l : pCorrect.keySet())
		{
			if (l.length() > longest)
				longest = l.length();
		}
		
		List<String> orderedKeys = getAttributeNamesSortedByMaxErrorCount();
		
		for (String ref : orderedKeys)
		{
			String padding = Helper.getNChars(longest - ref.length(), ' ');
			result += String.format("%s%s: %s (%d/%d)\n", padding, ref, Helper.formatDouble(pCorrect.get(ref)), 
					(int)matrixCounter.getCount(ref, ref), matrixCounter.getTotalCount(ref));
		}
		
		return result;
	}
		
	public String getAccuracyPrinted()
	{
		int totalCount = 0;
		for (String ref : matrixCounter.getAllAttributeNames())
		{
			for (String hyp : matrixCounter.getAllAttributeNames())
			{
				totalCount += getCount(ref, hyp);
			}
		}
		
		assert totalCount > 0;
		
		int correctCount = 0;
		for (String ref : matrixCounter.getAllAttributeNames())
		{
			correctCount += getCount(ref, ref);
		}
		assert correctCount <= totalCount;
		return String.format("Total: %s (%d/%d)\n", Helper.formatDouble(((double)correctCount)/totalCount),
				correctCount, totalCount);
	}
	
	/**
	 * 
	 * @return The total accuracy of the confusion matrix over all attributes.
	 */
	public double getAccuracy()
	{
		int totalCount = 0;
		for (String ref : matrixCounter.getAllAttributeNames())
		{
			for (String hyp : matrixCounter.getAllAttributeNames())
			{
				totalCount += getCount(ref, hyp);
			}
		}
		
		assert totalCount > 0;
		
		int correctCount = 0;
		for (String ref : matrixCounter.getAllAttributeNames())
		{
			correctCount += getCount(ref, ref);
		}
		assert correctCount <= totalCount;
		return ((double)correctCount)/totalCount;
	}
	
	/**
	 * Prints this confusion matrix in tab delimited format.
	 */
	@Override
	public String toString()
	{
		String result = "";
		//List<String> labels = getAttributeNamesSortedByMaxErrorCount();
		List<String> labels = Arrays.asList(matrixCounter.getAllAttributeNames().toArray(new String[0]));
		Collections.sort(labels);
		
		// Print column headers.
		result += "\t";
		for(String l : labels)
		{
			result += l + "\t";
		}
		result += "\n";
		
		// Find the length of the longest label.
		int longest = 0;
		for (String l : labels)
		{
			if (l.length() > longest)
				longest = l.length();
		}
		
		for(int row = 0; row < labels.size(); row++)
		{
			result += Helper.getNChars(longest - labels.get(row).length(), ' ');
			result += labels.get(row) + "\t";
			for(int col = 0; col < labels.size(); col++)
			{
				// Don't show correct counts.
				//if (row == col)
				//	result += "0" + "\t";
				//else
					result += (int)matrixCounter.getCount(labels.get(row), labels.get(col)) + "\t";
			}
			result += "\n";
		}
		
		return result;
	}
		
	public String getLabelName()
	{
		return labelName;
	}
	
	public void writeToFile(String csvFileName) throws IOException
	{
		File file = new File(csvFileName);
		
		if (!file.exists()) {
			file.createNewFile();
		}
		
		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
		
		bw.write(toString());
		
		bw.close();
	}
	
	private List<String> getAttributeNamesSortedByMaxErrorCount()
	{
		// Find the value of the largest cell for each row (label).
		Map<String, Integer> labelMaxes = new TreeMap<String, Integer>();
		for (String label : matrixCounter.getAllAttributeNames())
		{
			labelMaxes.put(label, matrixCounter.getMaxIgnoringCorrectCount(label));
		}
		
		// Sort labelMaxes by maxes (the counts).
		List<String> result = new ArrayList<String>();
		while(!labelMaxes.isEmpty())
		{
			String maxKey = Helper.argmax(labelMaxes);
			result.add(maxKey);
			labelMaxes.remove(maxKey);
		}
		
		return result;
	}
	
	private static class MatrixCounter
	{
		private Map<String, Map<String, Integer>> counts;
		
		public MatrixCounter()
		{
			counts = new TreeMap<String, Map<String, Integer>>();
		}
		
		public int getCount(String refName, String hypName)
		{
			Map<String, Integer> refMap = counts.get(refName);
			if (refMap == null)
				return 0;
			Integer result = refMap.get(hypName);
			if (result == null)
				return 0;
			else
				return result;
		}
		
		public void incrementCount(String refName, String hypName)
		{
			Map<String, Integer> refMap = counts.get(refName);
			if (refMap == null)
			{
				refMap = new TreeMap<String, Integer>();
				counts.put(refName, refMap);
			}
			if (refMap.containsKey(hypName))
			{
				refMap.put(hypName, refMap.get(hypName) + 1);
			}
			else
			{
				refMap.put(hypName, 1);
			}
		}
		
		public Set<String> getAllAttributeNames()
		{
			// I can't just return counts.keySet() because there may be some attribute values that
			// were predicted but never expected.
			Set<String> names = new HashSet<String>(counts.keySet());
			for (Map.Entry<String, Map<String, Integer>> entry : counts.entrySet())
			{
				names.addAll(entry.getValue().keySet());
			}
			return names;
		}
		
		/**
		 * Sums all of the counts along a row of the confusion matrix.
		 * @param refName The row to sum.
		 */
		public int getTotalCount(String refName)
		{
			Map<String, Integer> refMap = counts.get(refName);
			if (refMap == null)
				return 0;
			int result = 0;
			for (String hypName : refMap.keySet())
			{
				result += refMap.get(hypName);
			}
			return result;
		}
		
		/**
		 * Returns the highest count on a row of the confusion matrix.
		 * @param refName The attribute name of the row.
		 */
		public int getMaxIgnoringCorrectCount(String refName)
		{
			Integer maxCount = -1;
			Map<String, Integer> refMap = counts.get(refName);
			if (refMap == null)
			{
				// There happened to be no labels who's target value was refName.
				return 0;
			}
			for (String key : refMap.keySet())
			{
				if (refMap.get(key) > maxCount && !key.equals(refName))
				{
					maxCount = refMap.get(key);
				}
			}
			return maxCount;
		}
	}


}