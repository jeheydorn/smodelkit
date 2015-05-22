package smodelkit.util;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;

import org.json.simple.JSONArray;

import smodelkit.Vector;

public class Helper 
{
	private static DecimalFormat decimalFormat;
	
	static
	{
		resetDecimalFormat();
	}
	
	public static void setDecimalFormat(DecimalFormat format)
	{
		decimalFormat = format;
	}
	
	public static void resetDecimalFormat()
	{
		decimalFormat = new DecimalFormat("#.#####");
	}

	public static String formatDouble(double d)
	{
		return decimalFormat.format(d);
	}
	
	public static String formatDoubleList(List<Double> list)
	{
	   return formatDoubleList(list, decimalFormat);
	}

	public static String formatDoubleList(List<Double> list, DecimalFormat format)
	{
	    StringBuilder sb = new StringBuilder();
	    String sep = "";
	    for (Double d: list) 
	    {
	        sb.append(sep);
	        sb.append(format.format(d));
	        sep = " ";
	    }
	    return sb.toString();
	}

	public static String printArray(String title, double[] vect)
	{
		String result = title + ": ";
		result += arrayToString(vect);
		result += "\n";
		return result;
	}

	public static String arrayToString(double[] vect)
	{
		String result = "";
		for(int j = 0; j < vect.length; j++)
		{
			result += formatDouble(vect[j]);
			if (j + 1 < vect.length)
				result += " ";
		}
		return result;
	}

	public static String arrayToString2D(String title, String innerTitle, double[][] vect)
	{
		String result = title + ": \n";
		for(int i = 0; i < vect.length; i++)
		{
			result += printArray(innerTitle + " " + i, vect[i]) + "\n";
		}
		return result + "\n";
	}

	public static String arrayToString2D(String title, double[][] vect)
	{
		String result = title + ": \n";
		for(int i = 0; i < vect.length; i++)
		{
			result += arrayToString(vect[i]);
		}
		return result + "\n";
	}
	
	public static String printArray(Object[] array, String separator)
	{
		StringBuilder builder = new StringBuilder();
		for (int i = 0; i < array.length; i++)
		{
			builder.append(array[i]);
			if (i < array.length + 1)
				builder.append(separator);
		}
		return builder.toString();
	}
	
	public static String listToStringWithSeparator(List<? extends Object> list, String separator)
	{
		StringBuilder builder = new StringBuilder();
		for (int i = 0; i < list.size(); i++)
		{
			builder.append(list.get(i));
			if (i < list.size() - 1)
				builder.append(separator);
		}
		return builder.toString();
	}

	
	public static String readFile(String path) throws IOException 
	{
		if (path == null)
			throw new IllegalArgumentException("path cannot be null");
		Charset encoding = Charset.defaultCharset();
		byte[] encoded = Files.readAllBytes(Paths.get(path));
		return encoding.decode(ByteBuffer.wrap(encoded)).toString();
	}
	
	public static List<Double> toDoubleList(double[] array)
	{
		List<Double> result = new ArrayList<Double>(array.length);
		for (int i = 0; i < array.length; i++)
		{
			result.add(i, array[i]);
		}
		return result;
	}
	
	public static double[] toDoubleArray(List<Double> list)
	{
		double[] result = new double[list.size()];
		for (int i = 0; i < list.size(); i++)
		{
			result[i] = list.get(i);
		}
		return result;
	}
	
	public static int[] toIntArray(List<Integer> list)
	{
		int[] result = new int[list.size()];
		for (int i = 0; i < list.size(); i++)
		{
			result[i] = list.get(i);
		}
		return result;
	}
	
	public static long[] toLongArray(List<Long> list)
	{
		long[] result = new long[list.size()];
		for (int i = 0; i < list.size(); i++)
		{
			result[i] = list.get(i);
		}
		return result;
	}

	public static int[] JSONArrayToIntArray(JSONArray list)
	{
		int[] result = new int[list.size()];
		for (int i = 0; i < list.size(); i++)
		{
			result[i] = (int)(long)list.get(i);
		}
		return result;
	}

	public static double[] JSONArrayToDoubleArray(JSONArray list)
	{
		double[] result = new double[list.size()];
		for (int i = 0; i < list.size(); i++)
		{
			if (list.get(i) instanceof Long)
			{
				result[i] = ((Long)list.get(i)).doubleValue();
			}
			else
			{
				result[i] = (double)list.get(i);
			}
		}
		return result;
	}

	public static String[] JSONArrayToStringArray(JSONArray array)
	{
		String[] result = new String[array.size()];
		for (int i = 0; i < array.size(); i++)
		{
			result[i] = (String)array.get(i);
			
		}
		return result;
	}
	
	public static int indexOfMaxElementInRange(double[] array, int start, int length)
	{
		assert start + length <= array.length;
		if (length == 0)
			throw new IllegalArgumentException("length cannot be 0.");
		int maxIndex = start;
		for (int i = start + 1; i < start + length; i++)
		{
			if (array[i] > array[maxIndex])
			{
				maxIndex = i;
			}
		}
		return maxIndex;
	}

	public static int indexOfMaxElementInRange(Vector vector, int start, int length)
	{
		assert start + length <= vector.size();
		if (length == 0)
			throw new IllegalArgumentException("length cannot be 0.");
		int maxIndex = start;
		for (int i = start + 1; i < start + length; i++)
		{
			if (vector.get(i) > vector.get(maxIndex))
			{
				maxIndex = i;
			}
		}
		return maxIndex;
	}

	public static double max(double[] array)
	{
		if (array.length == 0)
			throw new IllegalArgumentException("length cannot be 0.");
		double maxElement = Double.NEGATIVE_INFINITY;
		for (double d : array)
		{
			if (d > maxElement)
				maxElement = d;
		}
		return maxElement;
	}
	
	public static double min(double[] array)
	{
		if (array.length == 0)
			throw new IllegalArgumentException("length cannot be 0.");
		double maxElement = Double.POSITIVE_INFINITY;
		for (double d : array)
		{
			if (d < maxElement)
				maxElement = d;
		}
		return maxElement;
	}
	
	public static double mean(double[] array)
	{
		if (array.length == 0)
			throw new IllegalArgumentException("length cannot be 0.");
		double sum = 0;
		for (double d : array)
		{
			sum += d;
		}
		return sum / array.length;
	}

	/**
	 * Returns an array of indexes into the given array, where the indexes are ordered such that
	 * they sort their targets from greatest to least.
	 * @param array
	 * @return
	 */
	public static Integer[] sortIndexesDescending(final double[] in)
	{
		Integer[] result = new Integer[in.length];
		
		for (int i = 0; i < result.length; i++)
			result[i] = i;
		
		Comparator<Integer> comparator = new Comparator<Integer>()
		{
			@Override
			public int compare(Integer i1, Integer i2) 
			{
				return Double.compare(in[i1], in[i2]) * -1;
			}
			
		};
		Arrays.sort(result, comparator);
		return result;
	}

	
	public static double maxElement(double[] d)
	{
		if (d.length == 0)
			throw new IllegalArgumentException("length cannot be 0.");
		double maxValue = d[0];
		for (int i = 1; i < d.length; i++)
		{
			if (d[i] > maxValue)
				maxValue = d[i];
		}
		return maxValue;
	}

	public static boolean isDouble(String str)
	{
		try
		{
			Double.parseDouble(str);
			return true;
		}
		catch(NumberFormatException ex)
		{
			return false;
		}
	}
	
	public static boolean isInteger(String str)
	{
		try
		{
			Integer.parseInt(str);
			return true;
		}
		catch(NumberFormatException ex)
		{
			return false;
		}
	}
		
	public static double[] concatArrays(double[] d1, double[] d2)
	{
		double[] result  = new double[d1.length + d2.length];
		for (int i = 0; i < d1.length; i++)
			result[i] = d1[i];
		for (int i = 0; i < d2.length; i++)
			result[i + d1.length] = d2[i];
		return result;
	}
	
	public static String[] concatArrays(String[] o1, String[] o2)
	{
		String[] result  = new String[o1.length + o2.length];
		for (int i = 0; i < o1.length; i++)
			result[i] = o1[i];
		for (int i = 0; i < o2.length; i++)
			result[i + o1.length] = o2[i];
		return result;
	}
	
	/**
	 * Returns the first line from the given file which contains the given content, or null if the
	 * content is not found.
	 */
	public static String getLineWithContent(File inputFile, String lineContent) throws IOException
	{
		try (BufferedReader reader = new BufferedReader(new FileReader(inputFile)))
		{
			String line;
			while((line = reader.readLine()) != null) 
			{
			    if (line.contains(lineContent))
			    {
			    	return line;
			    }
			}
		}
		return null;
	}
		
	public static String removeFileExtension(String filename)
	{
		// From http://stackoverflow.com/questions/924394/how-to-get-file-name-without-the-extension.
		return filename.replaceFirst("[.][^.]+$", "");
	}
	
	public static <K, V extends Comparable<V>> K argmax(Map<K, V> map)
	{
		Map.Entry<K, V> maxEntry = null;

		for (Map.Entry<K, V> entry : map.entrySet())
		{
		    if (maxEntry == null || entry.getValue().compareTo(maxEntry.getValue()) > 0)
		    {
		        maxEntry = entry;
		    }
		}
		return maxEntry.getKey();
	}

	public static <T extends Comparable<T>> int indexOfMaxElement(List<T> list)
	{
		if (list.isEmpty())
		{
			throw new IllegalArgumentException("There is no maximum element of an empty list.");
		}
		int maxIndex = 0;

		for (int i : new Range(1, list.size()))
		{
		    if (list.get(i).compareTo(list.get(maxIndex)) > 0)
		    {
		        maxIndex = i;
		    }
		}
		return maxIndex;
	}

	public static <T extends Comparable<T>> int indexOfMaxElement(double[] array)
	{
		if (array.length == 0)
		{
			throw new IllegalArgumentException("There is no maximum element of an empty list.");
		}
		int maxIndex = 0;

		for (int i : new Range(1, array.length))
		{
		    if (Double.compare(array[i], array[maxIndex]) > 0)
		    {
		        maxIndex = i;
		    }
		}
		return maxIndex;
	}

	public static <T> int indexOfMaxElement(List<T> list, Comparator<T> comparator)
	{
		if (list.isEmpty())
		{
			throw new IllegalArgumentException("There is no maximum element of an empty list.");
		}
		int maxIndex = 0;

		for (int i : new Range(1, list.size()))
		{
		    if (comparator.compare(list.get(i), list.get(maxIndex)) > 0)
		    {
		        maxIndex = i;
		    }
		}
		return maxIndex;
	}

	public static String getNChars(int n, char c)
	{
		StringBuffer outputBuffer = new StringBuffer(n);
		for (int i = 0; i < n; i++)
		{
		   outputBuffer.append(c);
		}
		return outputBuffer.toString();
	}
	
	/**
	 * Normalizes the given array such that the sum of its elements equals 1.
	 * All elements must be non-negative. If all elemnts are zero, then
	 * the given array is assigned a uniform distribution.
	 * @param array
	 */
	public static void normalize(double[] array)
	{
		double sum = 0;
		for (double d : array)
		{
			if (d < 0)
				throw new IllegalArgumentException("Elements cannot be negative. value: " + d);
			sum += d;
		}
		if (sum == 0)
		{
			double value = 1.0 / array.length;
			for (int i = 0; i < array.length; i++)
			{
				array[i] = value;
			}
			return;
		}
		if (Double.isNaN(sum))
		{
			throw new IllegalArgumentException("Sum is NaN. Did the array have unknown values?");
		}
		for (int i = 0; i < array.length; i++)
		{
			array[i] /= sum;
		}
	}
	
	public static <I, R> List<R> map(List<I> items, Function<I, R> fun)
	{
		List<R> result = new ArrayList<R>();
		for (I item : items)
			result.add(fun.apply(item));
		return result;
	}
		
	public static void processInParallel(List<Runnable> jobs)
	{
		List<Future<?>> futures = new ArrayList<Future<?>>();
		int threadsReserved = ThreadCounter.reserveThreadCount(jobs.size());
		int numThreads = Math.max(1, threadsReserved);
		ExecutorService exService = Executors.newFixedThreadPool(numThreads);
		try
		{
			for (Runnable job : jobs)
			{
				futures.add(exService.submit(job));
			}
	
			for (int i : new Range(jobs.size()))
			{
				try
				{
					futures.get(i).get();
				}
				catch(ExecutionException e)
				{
					throw new RuntimeException(e);
				}
				catch(InterruptedException e)
				{
					throw new RuntimeException(e);
				}
			}
		}
		finally
		{
			exService.shutdown();
			ThreadCounter.freeThreadCount(threadsReserved);
		}
	}
	
	public static <T> List<T> processInParallelAndGetResult(List<Callable<T>> jobs)
	{
		List<Future<T>> futures = new ArrayList<>();
		List<T> results = new ArrayList<>();
		int threadsReserved = ThreadCounter.reserveThreadCount(jobs.size());
		int numThreads = Math.max(1, threadsReserved);
		ExecutorService exService = Executors.newFixedThreadPool(numThreads);
		try
		{
			for (Callable<T> job : jobs)
			{
				futures.add(exService.submit(job));
			}
	
			for (int i : new Range(jobs.size()))
			{
				try
				{
					T result = futures.get(i).get();
					results.add(result);
				}
				catch(ExecutionException e)
				{
					throw new RuntimeException(e);
				}
				catch(InterruptedException e)
				{
					throw new RuntimeException(e);
				}
			}
		}
		finally
		{
			exService.shutdown();
			ThreadCounter.freeThreadCount(threadsReserved);
		}
		
		return results;
	}
	
	public static <T> List<T> iteratorToList(Iterator<T> iter)
	{
		ArrayList<T> result = new ArrayList<>();
		while(iter.hasNext())
			result.add(iter.next());
		return result;
	}
	
	/**
	 * Runs a command and waits for it to finish.
	 */
	public static void executeCommand(String command)
	{
		Process process;
		try
		{
			process = Runtime.getRuntime().exec(command);
		} catch (IOException e1)
		{
			throw new RuntimeException(e1);
		}
			
        // Read error stream
		StreamGobbler errorGobbler = new 
		StreamGobbler(process.getErrorStream(), "ERROR");            
		 
		 // Read output stream
		StreamGobbler outputGobbler = new 
		StreamGobbler(process.getInputStream(), "OUTPUT");
		     
		 // kick them off
		errorGobbler.start();
		outputGobbler.start();
		
		try
		{
			process.waitFor();
		}
		catch (InterruptedException e)
		{
			throw new RuntimeException();
		}
	}

	/**
	 * From http://www.javaworld.com/article/2071275/core-java/when-runtime-exec---won-t.html?page=2
	 * @author Michael C. Daconta 
	 * OK to use according to http://www.javaworld.com/article/2075891/java-app-dev/license-terms-for-java-code--javaworld.html
	 */
	private static class StreamGobbler extends Thread
	{
	    InputStream is;
	    String type;
	    
	    StreamGobbler(InputStream is, String type)
	    {
	        this.is = is;
	        this.type = type;
	    }
	    
	    public void run()
	    {
	        try
	        {
	            InputStreamReader isr = new InputStreamReader(is);
	            BufferedReader br = new BufferedReader(isr);
	            String line=null;
	            while ( (line = br.readLine()) != null)
	                System.out.println(type + ">" + line);    
	            } catch (IOException ioe)
	              {
	                ioe.printStackTrace();  
	              }
	    }
	}
	
	public static double findVariance(List<Double> values)
	{
		if (values.isEmpty())
			throw new IllegalArgumentException();
		
		double average = values.stream().mapToDouble(d -> d).average().getAsDouble();

		return values.stream().map(value 
				-> (average - value) * (average - value)).mapToDouble(d -> d).average().getAsDouble();
	}
	
	public static double findStandardDeviation(List<Double> values)
	{
		return Math.sqrt(findVariance(values));
	}
	
	/**
	 * Creates a deep copy of an object using serialization.
	 */
	public static Object deepCopy(Object toCopy)
	{
		ByteArrayOutputStream ostream = new ByteArrayOutputStream();
		byte[] storedObjectArray;
		{
			try (ObjectOutputStream p = new ObjectOutputStream(new BufferedOutputStream(ostream)))
			{
				p.writeObject(toCopy);
				p.flush();
			} catch (IOException e)
			{
				throw new RuntimeException(e);
			}
			storedObjectArray = ostream.toByteArray();
		}

		Object toReturn = null;
		try (ByteArrayInputStream istream = new ByteArrayInputStream(storedObjectArray))
		{
			ObjectInputStream p;
			p = new ObjectInputStream(new BufferedInputStream(istream));
			toReturn = p.readObject();
		} catch (IOException | ClassNotFoundException e)
		{
			throw new RuntimeException(e);
		}
		return toReturn;
	}
}
































