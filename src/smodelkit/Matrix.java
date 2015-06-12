package smodelkit;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import smodelkit.util.Counter;
import smodelkit.util.Helper;
import smodelkit.util.Pair;
import smodelkit.util.Range;
import smodelkit.util.Tuple2Comp;

/**
 * Represents a dataset. 
 *
 */
public class Matrix implements Serializable, Iterable<Vector>
{
	private static final long serialVersionUID = 1L;
	
	/**
	 * Determines if Vectors are created using doubles (true) or floats (false).
	 */
	public static boolean useDouble = true;

	String relationName;
	
	String comments;
	
	// Stores instance rows and their weights.
	ArrayList<Vector> data;

	// Meta-data
	ArrayList<String> attrNames;
	ArrayList<TreeMap<String, Integer>> strToEnum;
	ArrayList<TreeMap<Integer, String>> enumToStr;
	// This stores the number of columns that are used in each categorical distribution
	// when NominalToCategorical is used. This assumes all columns are either converted
	// from nominal values or are all real valued, but not mixed.
	List<Integer> numCatagoricalCols;
	/**
	 * This is the number of columns (from the right side) that are output columns.
	 * This is set using -c in the relation name. The default is 1.
	 */
	private int numLabelColumns;
	
	/**
	 *  Creates a 0x0 matrix. You should call loadARFF or setSize next.
	 */
	public Matrix()
	{
		relationName = null;
		data = new ArrayList<>();
		attrNames = new ArrayList<String>();
		strToEnum = new ArrayList<TreeMap<String, Integer>>();
		enumToStr = new ArrayList<TreeMap<Integer, String>>();
		numCatagoricalCols = new ArrayList<Integer>();
		numLabelColumns = 1;
	}

	/**
	 * Copy constructor. Everything is copied except the internal arrays in Vectors,
	 * which are immutable. 
	 * @param other
	 */
	public Matrix(Matrix other)
	{
		relationName = other.relationName;
		attrNames = new ArrayList<String>();
		strToEnum = new ArrayList<TreeMap<String, Integer>>();
		enumToStr = new ArrayList<TreeMap<Integer, String>>();
		for (int i = 0; i < other.cols(); i++)
		{
			attrNames.add(other.getAttrName(i));
			strToEnum.add(other.strToEnum.get(i));
			enumToStr.add(other.enumToStr.get(i));
		}
		numCatagoricalCols = new ArrayList<Integer>();
		for (int i : other.numCatagoricalCols)
		{
			numCatagoricalCols.add(i);
		}

		data = new ArrayList<>();
		for (int j = 0; j < other.rows(); j++)
		{
			addRow(other.row(j));
		}
		numLabelColumns = other.numLabelColumns;
	}

	/**
	 *  Copies the specified portion of that matrix into this matrix
	 */
	public Matrix(Matrix other, int rowStart, int colStart, int rowCount,
			int colCount)
	{
		attrNames = new ArrayList<String>();
		strToEnum = new ArrayList<TreeMap<String, Integer>>();
		enumToStr = new ArrayList<TreeMap<Integer, String>>();
		for (int i = 0; i < colCount; i++)
		{
			attrNames.add(other.getAttrName(colStart + i));
			strToEnum.add(other.strToEnum.get(colStart + i));
			enumToStr.add(other.enumToStr.get(colStart + i));
		}
		if (other.numCatagoricalCols.size() != 0)
			throw new UnsupportedOperationException("Categorical distributions are not supported when copying " +
					" a part of a matrix.");
		numCatagoricalCols = new ArrayList<Integer>();

		data = new ArrayList<>();
		for (int j = 0; j < rowCount; j++)
		{
			Vector rowSrc = other.row(rowStart + j);
			double[] rowDest = new double[colCount];
			for (int i = 0; i < colCount; i++)
				rowDest[i] = rowSrc.get(colStart + i);
			addRow(Vector.create(rowDest, rowSrc.getWeight()));
		}
		numLabelColumns = 0;
	}

	public void copyMetadata(Matrix other)
	{
		relationName = other.relationName;
		attrNames = new ArrayList<String>(other.attrNames);
		strToEnum = new ArrayList<TreeMap<String, Integer>>(other.strToEnum);
		enumToStr = new ArrayList<TreeMap<Integer, String>>(other.enumToStr);
		numCatagoricalCols = new ArrayList<Integer>(other.numCatagoricalCols);
		numLabelColumns = other.numLabelColumns;
	}
	
	/**
	 * Adds an empty  attribute column as the last column in this matrix. 
	 * By default this column is continuous. To make it nominal, add attribute
	 * values by calling addAttributeValue.
	 */
	public void addEmptyColumn(String attributeName)
	{		
		if (attrNames.contains(attributeName))
			throw new IllegalArgumentException("This matrix alread contains attribute name: " + attributeName);
		
		attrNames.add(attributeName);
		strToEnum.add(new TreeMap<>());
		enumToStr.add(new TreeMap<>());
	}
	
	/**
	 * Add an attribute value to the specified column.
	 * @param column
	 * @param attributeValueName
	 */
	public void addAttributeValue(int column, String attributeValueName)
	{		
		if (strToEnum.get(column).keySet().contains(attributeValueName))
			throw new IllegalArgumentException("Column " + column + " alread contains attribute value: " + attributeValueName);
		
		int largestAttrIndex = enumToStr.get(column).isEmpty() ? 0 : 
			Collections.max(enumToStr.get(column).keySet()) + 1;
		strToEnum.get(column).put(attributeValueName, largestAttrIndex);
		enumToStr.get(column).put(largestAttrIndex, attributeValueName);
	}
	
	public void addAttributeValueIfItDoesNotExist(int column, String attributeValueName)
	{
		if (strToEnum.get(column).keySet().contains(attributeValueName))
			return;
		
		int largestAttrIndex = enumToStr.get(column).isEmpty() ? 0 : 
			Collections.max(enumToStr.get(column).keySet()) + 1;
		strToEnum.get(column).put(attributeValueName, largestAttrIndex);
		enumToStr.get(column).put(largestAttrIndex, attributeValueName);		
	}

	public int getNumLabelColumns()
	{
		return numLabelColumns;
	}

	public void setNumLabelColumns(int numLabelColumns)
	{
		this.numLabelColumns = numLabelColumns;
	}
	
	/** Adds a copy of the specified portion of that matrix to this matrix
	 * 
	 * The number of columns that will be copied to this matrix is this.cols().
	 * 
	 * @param other
	 * @param rowStart
	 * @param colStart
	 * @param rowCount
	 * @throws Exception
	 */
	
	public void add(Matrix other, int rowStart, int colStart, int rowCount)
	{
		if (colStart + cols() > other.cols())
			throw new IllegalArgumentException("out of range");
		for (int i = 0; i < cols(); i++)
		{
			if (other.getValueCount(colStart + i) != getValueCount(i))
				throw new IllegalArgumentException("incompatible relations");
		}
		for (int j = 0; j < rowCount; j++)
		{
			Vector rowSrc = other.row(rowStart + j);
			double[] rowDest = new double[cols()];
			for (int i = 0; i < cols(); i++)
				rowDest[i] = rowSrc.get(colStart + i);
			addRow(Vector.create(rowDest, rowSrc.getWeight()));
		}
	}

	public void addRows(Matrix other, int num)
	{
		addRows(other, 0, num);
	}
			
	public void addRows(Matrix other, int start, int num)
	{
		for (int i = start; i < start + num; i++)
			addRow(other.row(i));
	}
	
	public void addRows(Matrix other, List<Integer> rowsToCopy)
	{
		for (int i : rowsToCopy)
		{
			addRow(other.row(i));
		}
	}
	
	public void removeRow(int row)
	{
		data.remove(row);
	}
	
	public void removeColumn(int colNumber)
	{
		if (numCatagoricalCols.size() != 0)
			throw new UnsupportedOperationException("Cannot remove columns from a matrix that has been filtered "
					+ "by NominalToCategorical.java");
		attrNames.remove(colNumber);
		strToEnum.remove(colNumber);
		enumToStr.remove(colNumber);

		for(int i = 0; i < rows(); i++)
		{
			data.get(i).remove(colNumber);
		}

		assert attrNames.size() == row(0).size();
	}
	
	/**
	 *  Copies num columns from that to this beginning at start. Meta-data is copied too.
	 *  
	 *  If this is empty, then instance weights are set to those from other. If this is not empty,
	 *  then instance weights are not changed.
	 */
	public void copyColumns(Matrix other, int start, int num)
	{		
		if (rows() > 0 && rows() != other.rows())
			throw new IllegalArgumentException("other must have the same number of rows as this.");
		if (start + num > other.cols())
		{
			throw new IllegalArgumentException("Index out of range.");
		}
		if (num == 0)
		{
			throw new IllegalArgumentException("Did you mean to copy 0 columns?");
		}
		
		addListRange(other.attrNames, attrNames, start, num);
		addListRange(other.strToEnum, strToEnum, start, num);
		addListRange(other.enumToStr, enumToStr, start, num);

		if (other.numCatagoricalCols.size() > 0)
		{
			// Make sure we are copying over a whole categorical distribution. Currently I do
			// not support copying more than one categorical distribution.
			int catStart = 0;
			int i = 0;
			while(catStart < start)
			{
				catStart += other.numCatagoricalCols.get(i);
				i++;
			}
			if (catStart != start)
				throw new IllegalArgumentException("Argument give for start is not at the beginning of a "
						+ "categorical distribution.");
			if (other.numCatagoricalCols.get(i) != num)
				throw new IllegalArgumentException("Given num=" + num + ", which is not the size of the categorical"
						+ " distribution (" + other.numCatagoricalCols.get(i) + ")." 
						+ "\"start\"=" + start + ".");
		
			numCatagoricalCols.add(other.numCatagoricalCols.get(i));
		}

		if (data.size() > 0)
		{
			for(int r = 0; r < other.rows(); r++)
			{				
				double[] toAdd = new double[num];
				for (int c = 0; c < toAdd.length; c++)
					toAdd[c] = other.row(r).get(start + c);
				row(r).addAll(Vector.create(toAdd));
			}	
		}
		else
		{
			// I don't have any rows.
			for(int r = 0; r < other.rows(); r++)
			{
				addRow(other.row(r).subVector(start, start + num));
			}				
		}
	}
	
	/**
	 * Creates a new matrix like this one but with only the specified columns.
	 * @return
	 */
	public Matrix selectColumns(List<Integer> columns)
	{
		Matrix result = new Matrix();
		for (int c : columns)
		{
			result.copyColumns(this, c, 1);
		}
		return result;
	}
	
	/**
	 *  Returns a copy of the specified columns of this matrix. Meta-data is copied too.
	 */
	public Matrix getColumns(int start, int num)
	{
		Matrix result = new Matrix();
		result.copyColumns(this, start, num);
		return result;
	}
	
	/**
	 * Returns the index of the column of the given attribute name, or -1 if
	 * it is not found.
	 */
	public int getAttributeColumnIndex(String attrName)
	{
		return attrNames.indexOf(attrName);
	}
	
	/**
	 * Gets the value from the specified row corresponding to the specified attribute.
	 */
	public double get(int rowNumber, String attrName)
	{
		int index = attrNames.indexOf(attrName);
		if (index == -1)
			throw new IllegalArgumentException("Attribute name \"" + attrName + "\" not found.");
		return row(rowNumber).get(index);
	}
	
	public Matrix getColumnsIgnoringFilteredCatagoricalColumns(int start, int num)
	{
		Matrix result = new Matrix();
		this.numCatagoricalCols = new ArrayList<>();
		result.copyColumns(this, start, num);
		return result;
	}

	
	/**
	 * Adds num elements to the end of dest, starting with source.get(start). 
	 */
	private <T> void addListRange(List<T> source, List<T> dest, int start, int num)
	{
		for (int i = start; i < num + start; i++)
		{
			dest.add(source.get(i));
		}
	}

	/*
	 * Adds a copy of the given vector to this datset.
	 */
	public void addRow(Vector v)
	{
		v = Vector.create(v);
		// Verify the given row.
		if (v.size() != cols())
			throw new IllegalArgumentException("The given row is not the expected size for this dataset.");
		for (int i : new Range(v.size()))
		{
			if (!isContinuous(i))
			{
				if (v.get(i) < 0)
					throw new IllegalArgumentException("Nominal values cannot be negative.");
				if (v.get(i) >= getValueCount(i) && !Vector.isUnknown(v.get(i)))
				{
					throw new IllegalArgumentException("Nominal value is out of range.");
				}
			}
		}
		
		data.add(v);
	}
		
	public void checkCompatibility(Matrix other)
	{
		int c = cols();
		if(other.cols() != c)
			throw new IllegalArgumentException("Matrices have different number of columns");
		for(int i = 0; i < c; i++)
		{
			if(getValueCount(i) != other.getValueCount(i))
				throw new IllegalArgumentException("Column " + i + " has mis-matching number of values");
		}
	}

	public void clear()
	{
		attrNames.clear();
		strToEnum.clear();
		enumToStr.clear();
		data.clear();
	}
	
	/**
	 * Resizes this matrix (and sets all attributes to be continuous, with instance weights of 1)
	 */
	public void setSize(int rows, int cols)
	{
		data = new ArrayList<>();
		for (int j = 0; j < rows; j++)
		{
			double[] row = new double[cols];
			addRow(Vector.create(row, 1.0));
		}
		attrNames = new ArrayList<String>();
		strToEnum = new ArrayList<TreeMap<String, Integer>>();
		enumToStr = new ArrayList<TreeMap<Integer, String>>();
		for (int i = 0; i < cols; i++)
		{
			attrNames.add("");
			strToEnum.add(new TreeMap<String, Integer>());
			enumToStr.add(new TreeMap<Integer, String>());
		}
	}

	/**
	 *  Loads from an ARFF file
	 * @param filename
	 * @param loadComments If true, any comments before the relation name will be loaded.
	 */
	public void loadFromArffFile(String filename, boolean loadComments, int maxRows)
	{
		try (Scanner s = new Scanner(new File(filename)))
		{
			loadArff(s, loadComments, maxRows);
		} 
		catch (FileNotFoundException e)
		{
			throw new RuntimeException(e);
		}
	}

	/**
	 *  Loads from an ARFF file
	 * @param filename
	 */
	public void loadFromArffFile(String filename)
	{
		loadFromArffFile(filename, false, Integer.MAX_VALUE);
	}

	/**
	 *  Loads from a string containing data in the arff format.
	 * @param filename
	 * @throws FileNotFoundException
	 */
	public void loadFromArffString(String content)
	{
		try (Scanner s = new Scanner(content))
		{
			loadArff(s, false, Integer.MAX_VALUE);
		}
	}
	
	/**
	 *  Loads from an ARFF file
	 * @param filename
	 * @throws FileNotFoundException
	 */
	private void loadArff(Scanner s, boolean loadComments, int maxRows)
	{
		data = new ArrayList<>();
		attrNames = new ArrayList<String>();
		strToEnum = new ArrayList<TreeMap<String, Integer>>();
		enumToStr = new ArrayList<TreeMap<Integer, String>>();
		boolean READDATA = false;
		StringBuilder commentsBuilder = new StringBuilder();
		while (s.hasNext())
		{
			String line = s.nextLine().trim();
			if (line.length() > 0)
			{
				if (line.charAt(0) == '%')
				{
					if (loadComments)
					{
						commentsBuilder.append(line);
						commentsBuilder.append("\n");
					}
				}
				else
				{
					if (!READDATA)
					{
	
						try (Scanner t = new Scanner(line))
						{
							String firstToken = t.next().toUpperCase();
		
							if (firstToken.equals("@RELATION"))
							{
								QuoteParser u = new QuoteParser(line);
								u.next();
								relationName = u.next().replace("'", "").replace("\"", "");
																
								if (relationName.toUpperCase().contains("-C"))
								{
									QuoteParser p = new QuoteParser(relationName);
									while (!p.next().toUpperCase().equals("-C"))
									{
										
									}
									String labelColsStr = p.next();
									
									// Get the number of label columns.
									numLabelColumns = Integer.parseInt(labelColsStr);
									if (numLabelColumns > 0)
										throw new IllegalArgumentException("Label columns at the beginning of "
												+ "a dataset are not supported, so arguments given to -c in"
												+ " arff files must be negative. This is necessary to make dataset compatible"
												+ " with Meka.");
									// I need to be able to add a negative sign, and ignore it, to make my datasets compatible with Meka.
									numLabelColumns = Math.abs(numLabelColumns);
								}
								
								if (relationName.contains(":"))
								{
									relationName = relationName.split(":")[0];
								}
								
								t.nextLine();
							}
		
							if (firstToken.equals("@ATTRIBUTE"))
							{
								TreeMap<String, Integer> ste = new TreeMap<String, Integer>();
								strToEnum.add(ste);
								TreeMap<Integer, String> ets = new TreeMap<Integer, String>();
								enumToStr.add(ets);
		
								QuoteParser parser = new QuoteParser(line);
								parser.next();
								String attributeName = parser.next();
								if (attributeName.equals("?"))
									throw new IllegalArgumentException("\"?\" is a reserved token. Found in line: " + line);
								attrNames.add(attributeName);
	
								int vals = 0;
								String type = parser.next().toUpperCase();
								if (type.equals("REAL")
										|| type.equals("CONTINUOUS")
										|| type.equals("INTEGER")
										|| type.equals("NUMERIC"))
								{
								} 
								else
								{
									if (!line.contains("{"))
										throw new RuntimeException("Missing \"{\" from line: " + line );
									if (!line.contains("}"))
										throw new RuntimeException("Missing \"}\"");
									String values = line.substring(
											line.indexOf("{") + 1,
											line.indexOf("}"));
									try (Scanner v = new Scanner(values))
									{
										v.useDelimiter(",");
										while (v.hasNext())
										{
											String value = v.next().trim();
											if (value.length() > 0)
											{
												if (value.equals("?"))
													throw new IllegalArgumentException("\"?\" is a reserved token. Found in line: " + line);
												ste.put(value, new Integer(
														vals));
												ets.put(new Integer(vals),
														value);
												vals++;
											}
										}
									}
								}
							}
							if (firstToken.equals("@DATA"))
							{
								READDATA = true;
							}
						}
					} 
					else
					{
						if (rows() < maxRows)
							loadDataRow(line, Collections.emptyList());
						else
							break;
					}
				}
			}
		}
		comments = commentsBuilder.toString();
		validate();
	}
		
	/**
	 * Throws an exception if this matrix is not valid.
	 */
	private void validate()
	{
		// Check for duplicate attribute names.
		Set<String> prev = new TreeSet<>();
		int i = 0;
		for (String name : attrNames)
		{
			if (prev.contains(name))
				throw new IllegalArgumentException("Dupilcate attribute names are not allowed in arff"
						+ " format. Duplicate name: " + name + ", index: " + i);
			prev.add(name);
			i++;
		}
	}
	

	private void loadDataRow(String line, List<Integer> ignoredColumns)
	{
		double[] newRow = new double[cols()];
		double instanceWeight = 1.0;
		
		// There are 2 ways to store a data row: sparse or not sparse. The non-sparse way stores
		// all values separated by commas.
		// The sparse way stores key value pairs separate by commas, where they key is the index
		// of the attribute, and the value is the value it has. Indexes start at 0. If an attribute
		// does not have a value specified, it will be zero, or the first nominal value.
		
		line = line.trim();
		if (line.startsWith("{") && line.endsWith("}"))
		{
			// Data is stored in key value pairs.
						
			try (Scanner t = new Scanner(line))
			{
				t.useDelimiter(",");
				boolean atLeastOnePieceOfDataRead = false;
				while (t.hasNext())
				{
					String next = t.next().trim();
					if (!t.hasNext() && next.startsWith("{") && next.endsWith("}") && atLeastOnePieceOfDataRead)
					{
						instanceWeight = parseInstanceWeight(next, line);
						continue;
					}
								
					// Remove the curly brackets if present.
					if (next.charAt(0) == '{')
						next = next.substring(1, next.length());
					if (next.charAt(next.length() - 1) == '}')
						next = next.substring(0, next.length() - 1);
					
					// Check for the case where a row is just {}, verses has empty key-value pairs such as {,}. 
					if (next.trim().isEmpty())
					{
						if (line.contains(","))
							throw new IllegalArgumentException("Value \"" + next + "\" does not specify a key-value pair. Line: " + line);
						else
							break;
					}
					
					
					String[] parts = next.split(" ");
					if (parts.length != 2)
						throw new IllegalArgumentException("Value \"" + next + "\" does not specify a key-value pair. Line: " + line);
					int attrIndex;
					try
					{
						attrIndex = Integer.parseInt(parts[0]);
					}
					catch(NumberFormatException e)
					{
						throw new IllegalArgumentException("Cannot parse attribute index \"" + parts[0] + "\" from line: " + line);
					}
					
					if (attrIndex >= strToEnum.size())
					{
						throw new IllegalArgumentException("Index \"" + attrIndex + "\" is out of range in line: " + line);
					}
					
					if (isContinuous(attrIndex))
					{
						try
						{
							newRow[attrIndex] = Double.parseDouble(parts[1]);
						}
						catch(NumberFormatException e)
						{
							throw new IllegalArgumentException("Expected a continuous value, but got \"" + parts[1] + "\" in line: " + line);
						}
					}
					else
					{
						Integer attrValueAsInteger = strToEnum.get(attrIndex).get(parts[1]);
						if (attrValueAsInteger == null)
						{
							throw new IllegalArgumentException("Unrecognized attribute value \"" + parts[1] 
									+ "\" for attribute \"" + getAttrName(attrIndex) + "\" in line: " + line);
						}
						double doubleValue = (int)attrValueAsInteger;
						if (doubleValue == -1)
						{
							throw new IllegalArgumentException("Error parsing the value '" + parts[1] + "' on line: " + line);
						}
						if (newRow[attrIndex] != 0)
						{
							throw new IllegalArgumentException("Attribute " + attrIndex + " is specified multiple times in line: " + line);
						}
						newRow[attrIndex] = doubleValue;
					}
					
					atLeastOnePieceOfDataRead = true;
				}
			}
		}
		else
		{
			// Data is stored with every value specified separated by commas.
			
			int curPos = 0;
			int curPosInAttributeNames = 0;

			try (Scanner t = new Scanner(line))
			{
				t.useDelimiter(",");
				while (t.hasNext())
				{
					String textValue = t.next().trim();
									
					if (textValue.isEmpty())
						throw new RuntimeException("Line contains empty string in column " + curPos + ": " + line);
					
					if (!t.hasNext() && textValue.startsWith("{") && textValue.endsWith("}"))
					{
						instanceWeight = parseInstanceWeight(textValue, line);
						continue;
					}

					if (ignoredColumns.contains(curPos))
					{
						curPos++;
						continue;
					}
	
					double doubleValue;
					int vals;
					try
					{
						vals = enumToStr.get(curPosInAttributeNames).size();
					}
					catch(IndexOutOfBoundsException e)
					{
						throw new IllegalArgumentException("The given line has more entries than their are attributes. Line: " + line);
					}
	
					// Missing instances appear in the dataset
					// as a double defined in Vector.getUnknownValue().
					if (textValue.equals("?"))
					{
						doubleValue = Vector.getUnknownValue();
 					}
					// Continuous values appear in the instance
					// vector as they are
					else if (vals == 0)
					{
						doubleValue = Double
								.parseDouble(textValue);
					}
					// Discrete values appear as an index to the "name"
					// of that value in the "attributeValue" structure
					else
					{
						if (!strToEnum.get(curPosInAttributeNames).containsKey(textValue))
						{
							throw new RuntimeException(String.format(
									"Unknown attribute value \"%s\" for attribute \"%s\" in line:\n%s",
									textValue, getAttrName(curPosInAttributeNames), line));
						}
						doubleValue = strToEnum.get(curPosInAttributeNames)
								.get(textValue);
						if (doubleValue == -1)
						{
							throw new RuntimeException(
									"Error parsing the value '"
											+ textValue
											+ "' on line: "
											+ line);
						}
					}
	
					newRow[curPosInAttributeNames] = doubleValue;
					curPos++;
					curPosInAttributeNames++;
				}
			} 
		}
		addRow(Vector.create(newRow, instanceWeight));

	}
	
	/**
	 * Parses an instance weight from between { }.
	 * @param line For debugging only.
	 */
	private double parseInstanceWeight(String str, String line)
	{
		try
		{
			return Double.parseDouble(str.substring(1, str.length() - 1));
		}
		catch(NumberFormatException e)
		{
			throw new NumberFormatException("Unable to parse instance weight in line: " + line);
		}		
	}
	
	/**
	 *  Loads from a .names and .data or .test file.
	 * @throws FileNotFoundException
	 */
	public void loadFromNamesFormat(String namesFilename, String dataFilename) throws FileNotFoundException
	{
		List<Integer> ignoredColumns = loadNamesFile(namesFilename);
		
		loadDataFile(dataFilename, ignoredColumns);

		validate();
	}
	
	/**
	 * @return Columns that should be ignored when reading the data file.
	 * @throws FileNotFoundException
	 */
	private List<Integer> loadNamesFile(String namesFilename) throws FileNotFoundException
	{
		data = new ArrayList<>();
		attrNames = new ArrayList<String>();
		strToEnum = new ArrayList<TreeMap<String, Integer>>();
		enumToStr = new ArrayList<TreeMap<Integer, String>>();
		
		ArrayList<String> output_attr_name = new ArrayList<String>();
		ArrayList<TreeMap<String, Integer>> output_str_to_enum = new ArrayList<TreeMap<String, Integer>>();
		ArrayList<TreeMap<Integer, String>> output_enum_to_str = new ArrayList<TreeMap<Integer, String>>();
		int outputCols = 1;
		int lineNumber = 0;
		List<Integer> ignoredColumns = new ArrayList<Integer>();
		try (Scanner s = new Scanner(new File(namesFilename)))
		{
			while (s.hasNext())
			{
				lineNumber++;
				String line = s.nextLine();
				line = line.replaceAll("\\s|\\.", "");
				if (line.isEmpty())
				continue;
				
				// Remove comments.
				int commentStartIndex = line.indexOf('|');
				if (commentStartIndex != -1)
					line = line.substring(0, commentStartIndex);
				
				if (lineNumber == 1)
				{
					if (Helper.isInteger(line))
					{
						// This dataset has multiple target classes.
						outputCols = Integer.parseInt(line);
					}
				}
				else if (output_attr_name.size() < outputCols)
				{
					// Load the output column meta-data.
					String colName = "class" + (output_attr_name.size() + 1);
					output_attr_name.add(colName);
					TreeMap<String, Integer> ste = new TreeMap<String, Integer>();
					TreeMap<Integer, String> ets = new TreeMap<Integer, String>();
					String[] attrValues = line.split(",");
					for (int i = 0; i < attrValues.length; i++)
					{
						ste.put(attrValues[i], i);
						ets.put(i, attrValues[i]);
					}
					
					output_str_to_enum.add(ste);
					output_enum_to_str.add(ets);
				}
				else
				{
					// Load feature meta-data.
					String[] parts = line.split(":");
					if (parts.length != 2)
						throw new IllegalArgumentException("Expected exactly 1 ':' in the line: " + line);
					
					if (parts[1].equals("ignore"))
					{
						ignoredColumns.add(attrNames.size() + ignoredColumns.size());
					}
					else if (parts[1].equals("continuous"))
					{
						attrNames.add(parts[0]);	
						strToEnum.add(new TreeMap<String, Integer>());
						enumToStr.add(new TreeMap<Integer, String>());
					}
					else
					{				
						attrNames.add(parts[0]);
						
						// Get the attribute names.
						TreeMap<String, Integer> ste = new TreeMap<String, Integer>();
						TreeMap<Integer, String> ets = new TreeMap<Integer, String>();
						String[] attrValues = parts[1].split(",");
						for (int i = 0; i < attrValues.length; i++)
						{
							ste.put(attrValues[i], i);
							ets.put(i, attrValues[i]);
						}
						
						strToEnum.add(ste);
						enumToStr.add(ets);
					}
				}
			}
		}
		
		attrNames.addAll(output_attr_name);
		strToEnum.addAll(output_str_to_enum);
		enumToStr.addAll(output_enum_to_str);
		
		return ignoredColumns;
	}
	
	private void loadDataFile(String dataFilename, List<Integer> ignoredColumns) throws FileNotFoundException
	{
		try (Scanner s = new Scanner(new File(dataFilename)))
		{
			while (s.hasNext())
			{
				String line = s.next();
				
				// Remove this temp code when done matching Xinchuan's results. 
//				// This is for extracting Xinchuan's QT data.
//				double qt;
//				if (dataFilename.endsWith(".test"))
//					 qt = 1.0;
//				else
//					qt = Double.NEGATIVE_INFINITY;
//				String[] parts = line.split(",");
//				if (Double.parseDouble(parts[parts.length - 1]) >= qt)
//				{
//					parts = Arrays.copyOf(parts, parts.length -1);
//					line = StringUtils.join(parts, ",");
//					loadDataRow(line.replace(", " + parts[parts.length - 1], ""), ignoredColumns);
//				}
				
				loadDataRow(line, ignoredColumns);
			}
		}
	}

	// Returns the number of rows in the matrix
	public int rows()
	{
		return data.size();
	}

	// Returns the number of columns (or attributes) in the matrix
	public int cols()
	{
		return attrNames.size();
	}

	/**
	 * Returns the specified row. The values in the result must not be changed because
	 * an array of values for an instance might be shared by other parts of the code.
	 * @param r The index of the row to return.
	 */
	public Vector row(int r)
	{
		return data.get(r);
	}

	// Returns the name of the specified attribute
	public String getAttrName(int col)
	{
		return attrNames.get(col);
	}

	// Returns the name of the specified value
	public String getAttrValueName(int attr, int val)
	{
		String result = enumToStr.get(attr).get(val);
		if (result == null)
			throw new IndexOutOfBoundsException(
					String.format("Attribute \"%s\" does not have the value %s.", getAttrName(attr), val));
		return result;
	}
	
	/**
	 * Returns the index of the specified attribute name in the specified attribute column.
	 */
	public int getAttrValueIndex(int attr, String attrValueName)
	{
		return strToEnum.get(attr).get(attrValueName);
	}

	/**
	 * Returns the number of values associated with the specified attribute (or column)
	 * 0=continuous, 2=binary, 3=trinary, etc.
	 */
	public int getValueCount(int col)
	{
		return enumToStr.get(col).size();
	}
	
	public boolean isContinuous(int column) 
	{ 
		return getValueCount(column) == 0; 
	}
	
	public int countValues(int column, double value)
	{
		int count = 0;
		for (int r = 0; r < rows(); r++)
		{
			if (row(r).get(column) == value)
				count++;
		}
		return count;
	}
	
	/**
	 *  Shuffles the row order
	 */
	public void shuffle(Random rand)
	{
		for (int n = rows(); n > 0; n--)
		{
			int i = rand.nextInt(n);
			Vector tmp = data.get(n - 1);
			data.set(n - 1, data.get(i));
			data.set(i, tmp);
		}
	}

	public void shuffle(Random rand, Matrix buddy)
	{
		for (int n = rows(); n > 0; n--)
		{
			int i = rand.nextInt(n);

			Vector tmp = data.get(n - 1);
			data.set(n - 1, data.get(i));
			data.set(i, tmp);

			tmp = buddy.data.get(n - 1);
			buddy.data.set(n - 1, buddy.data.get(i));
			buddy.data.set(i, tmp);
		}
	}

	// Returns the mean of the specified column
	public double findMean(int col)
	{
		double sum = 0;
		int count = 0;
		for (int i = 0; i < rows(); i++)
		{
			double v = row(i).get(col);
			if (!Vector.isUnknown(v))
			{
				sum += v;
				count++;
			}
		}
		return sum / count;
	}

	// Returns the min value in the specified column
	public double findMin(int col)
	{
		double m = Vector.getUnknownValue();
		for (int i = 0; i < rows(); i++)
		{
			double v = row(i).get(col);
			if (!Vector.isUnknown(v))
			{
				if (Vector.isUnknown(m) || v < m)
					m = v;
			}
		}
		return m;
	}

	// Returns the max value in the specified column
	public double findMax(int col)
	{
		double m = Vector.getUnknownValue();
		for (int i = 0; i < rows(); i++)
		{
			double v = row(i).get(col);
			if (!Vector.isUnknown(v))
			{
				if (Vector.isUnknown(m) || v > m)
					m = v;
			}
		}
		return m;
	}

	/**
	 * Returns the most common value in the specified column
	 */
	public double findMode(int col)
	{
		TreeMap<Double, Integer> tm = new TreeMap<Double, Integer>();
		for (int i = 0; i < rows(); i++)
		{
			double v = row(i).get(col);
			if (!Vector.isUnknown(v))
			{
				Integer count = tm.get(v);
				if (count == null)
					tm.put(v, new Integer(1));
				else
					tm.put(v, new Integer(count.intValue() + 1));
			}
		}
		int maxCount = 0;
		double val = Vector.getUnknownValue();
		Iterator<Entry<Double, Integer>> it = tm.entrySet().iterator();
		while (it.hasNext())
		{
			Entry<Double, Integer> e = it.next();
			if (e.getValue() > maxCount)
			{
				maxCount = e.getValue();
				val = e.getKey();
			}
		}
		return val;
	}
	
	/**
	 * Returns a copy of all of the columns in this matrix which correspond to the nominal class
	 * specified by labelClass. This assumes that the NominalToCatagorical filter was used to
	 * create this matrix.
	 */
	public Matrix getCategoricalizedLabelCols(int labelClass)
	{
		if (labelClass > numCatagoricalCols.size())
			throw new IllegalArgumentException("labelClass out of range");
		
		int start = 0;
		for (int i = 0; i < labelClass; i++)
		{
			start += numCatagoricalCols.get(i);
		}
		
		Matrix result = new Matrix();
		result.copyColumns(this, start, numCatagoricalCols.get(labelClass));
		
		return result;		
	}
	
	/**
	 * Finds all column number in this matrix corresponding to the nominal value which
	 * col is a column from. 
	 * @return A pair containing the first (inclusive) and last (exclusive) columns.
	 */
	public Tuple2Comp<Integer, Integer> getColumnsCorespondingToFilteredNominalLabelInColumn(int col)
	{
		// Find the column number where the filtered nominal value starts.
		int first = 0;
		int i = 0;
		for (; first + numCatagoricalCols.get(i) < col; i++)
		{
			first += numCatagoricalCols.get(i);
		}
		
		return new Tuple2Comp<>(first, first + numCatagoricalCols.get(i));
	}
	
	public List<Integer> getNumCatagoricalCols() { return Collections.unmodifiableList(numCatagoricalCols); }

	public void setNumCatagoricalCols(List<Integer> numCatagoricalCols) 
	{
		this.numCatagoricalCols = new ArrayList<Integer>(numCatagoricalCols); 
	}

	/**
	 * If this matrix has been filtered using NominalToCatagorical, this will return the number of
	 * nominal column in the original matrix (before filtering). Otherwise this will be 0.
	 */
	public int getFilteredNominalColsTotal() { return numCatagoricalCols.size(); }
	
	/**
	 * Counts the number of nominal columns in this matrix. If the filter NominalToCatagorical has been
	 * applied, this should be 0.
	 * @return
	 */
	public int countNominalCols()
	{
		int count = 0;
		for (int i = 0; i < cols(); i++)
		{
			if (!isContinuous(i))
			{
				count++;
			}
		}
		return count;
	}
	
    public boolean hasNominalCols()
    {
    	for (int i = 0; i < cols(); i++)
    	{
    		if (!isContinuous(i))
    			return true;
    	}
    	return false;
    }
    
    public boolean hasContinuousCols()
    {
    	for (int i = 0; i < cols(); i++)
    	{
    		if (isContinuous(i))
    			return true;
    	}
    	return false;
    	
    }
            
    public boolean containsUnknowns()
    {
    	for (Vector vec : this)
    	{
    		for (int d = 0; d < vec.size(); d++)
    		{
    			if (Vector.isUnknown(vec.get(d)))
    				return true;
    		}
    	}
    	return false;
    }
    
    public String getRelationName()
    {
    	return relationName;
    }
    
    public void setRelationName(String name)
    {
    	relationName = name;
    }
   
    public String getComments()
    {
    	return comments;
    }
    
    public void setComments(String comments)
    {
    	this.comments = comments;
    }
    
        
    /**
     * Returns a new matrix in which classes are balanced by oversampling
     * the rows corresponding to under-represented classes.
     * 
     * This will give strange results if the class (last column) is real
     * valued and takes on many values.
     */
	public Matrix oversample(Random rand)
	{
		if (numLabelColumns > 1)
			throw new UnsupportedOperationException("Oversampling of mutli-dimensional datasets is not"
					+ " supported.");
		
		Counter<Double> counter = new Counter<Double>();
		for (int r : new Range(this.rows()))
		{
			counter.increment(row(r).get(this.cols() - 1));
		}
		
		Matrix result = new Matrix(this);

		int maxCount = counter.maxCount();
		
		// For each label value in the counter:
		for (Double output : counter.keySet())
		{
			//	Find all rows with that label value.
			List<Vector> rowsWithOutput = new ArrayList<>();
			for (int r : new Range(this.rows()))
			{
				if (row(r).get(this.cols() - 1) == output)
				{
					rowsWithOutput.add(data.get(r));
				}
			}
			
			//  Sample from those rows until the count equals maxCount.
			
			while(counter.getCount(output) < maxCount)
			{
				Vector instanceAndWeight = rowsWithOutput.get(rand.nextInt(rowsWithOutput.size()));
				result.addRow(instanceAndWeight);
				counter.increment(output);
			}
		}
		return result;
	}
	
	/**
	 * Splits the inputs and labels into new Matrixes.
	 * @return The first element is the inputs. The second is the labels.
	 */
	public Pair<Matrix> splitInputsAndLabels()
	{
		Matrix inputs = new Matrix(this, 0, 0, this.rows(), this.cols() - this.getNumLabelColumns());
		Matrix labels = new Matrix(this, 0, this.cols() - this.getNumLabelColumns(), this.rows(),
				this.getNumLabelColumns());
		return new Pair<>(inputs, labels);
	}


    @Override
    public String toString()
	{ 		
    	StringBuilder result = metaDataToString();
		result.append("@DATA\n");
		for (int i = 0; i < rows(); i++)
		{
			result.append(rowToString(i));
			result.append("\n");
		}
		return result.toString();
	}

    /**
     * Converts this matrix to a string using a sparse format for attribute values.
     * @return
     */
    public String toStringSparse()
	{
    	StringBuilder result = metaDataToString();
 		result.append("@DATA\n");
		for (int i = 0; i < rows(); i++)
		{
			result.append(rowToStringSparse(row(i)));
			result.append("\n");
		}
		return result.toString();
	}
    
    private StringBuilder metaDataToString()
    {
    	StringBuilder result = new StringBuilder();
    	if (comments != null && !comments.isEmpty())
    	{
    		result.append(comments);
    		result.append("\n");
    	}
    	if (numLabelColumns > 1)
    	{
    		result.append(String.format("@RELATION '%s: -c %d '\n", relationName, -numLabelColumns));
    	}
    	else
    	{
    		result.append(String.format("@RELATION %s\n", relationName));
    	}
    	result.append("\n");
		for (int i = 0; i < attrNames.size(); i++)
		{
			if (Helper.iteratorToList(new QuoteParser(attrNames.get(i))).size() > 1)
			{
				// This attribute name will be parsed into multiple tokens, meaning it has unqoated white space.
				result.append("@ATTRIBUTE \"" + attrNames.get(i) + "\"\t");
			}
			else
			{
				result.append("@ATTRIBUTE " + attrNames.get(i) + "\t");
			}
			
			int vals = getValueCount(i);
			if (vals == 0)
				result.append("NUMERIC\n");
			else
			{
				result.append("{");
				for (int j = 0; j < vals; j++)
				{
					if (j > 0)
						result.append(", ");
					result.append(enumToStr.get(i).get(j));
				}
				result.append("}\n");
			}
		}
    	result.append("\n");
    	return result;
    }

    public String rowToString(int rowNumber)
    {
    	return rowToString(row(rowNumber));
    }
    
    public String rowToString(Vector rowToPrint)
    {
       	StringBuilder result = new StringBuilder();
		for (int j = 0; j < rowToPrint.size(); j++)
		{
			if (j > 0)
				result.append(",");
			if (getValueCount(j) == 0)
			{
				if (Vector.isUnknown(rowToPrint.get(j)))
				{
					result.append("?");
				}
				else
				{
					if ((int)rowToPrint.get(j) == rowToPrint.get(j))
						result.append((int)rowToPrint.get(j));
					else
						result.append(rowToPrint.get(j));
				}
			}
			else
			{
				String valueName = enumToStr.get(j).get((int) rowToPrint.get(j));
				if (valueName == null)
				{
					if (Vector.isUnknown(rowToPrint.get(j)))
						valueName = "?";
					else
						throw new IllegalArgumentException("Unknown value " + rowToPrint.get(j) 
								+ " for attribute with index " + j);
				}
				result.append(valueName);
			}
		}
		
		if (rowToPrint.getWeight() != 1.0)
		{
			result.append(", {");
			result.append(rowToPrint.getWeight());
			result.append("}");
		}
		return result.toString();    	
    }
    
    public String rowToStringSparse(Vector rowToPrint)
    {
    	StringBuilder result = new StringBuilder();
    	result.append("{");
    	boolean aValueHasBeenWritten = false;
		for (int j = 0; j < rowToPrint.size(); j++)
		{
			// Only store values which are not 0 since 0 is the default when loading a sparse matrix.
			if (rowToPrint.get(j) != 0)
			{
				if (aValueHasBeenWritten)
					result.append(",");
				else
					aValueHasBeenWritten = true;
				result.append(j + " ");
				if (getValueCount(j) == 0)
				{
					if (Vector.isUnknown(rowToPrint.get(j)))
					{
						result.append("?");
					}
					else
					{
						if ((int)rowToPrint.get(j) == rowToPrint.get(j))
							result.append((int)rowToPrint.get(j));
						else
							result.append(rowToPrint.get(j));	
					}
				}
				else
				{
					String valueName = enumToStr.get(j).get((int) rowToPrint.get(j));
					valueName = valueName == null ? "?" : valueName;
					result.append(valueName);
				}
			}
		}
		result.append("}");
		
		if (rowToPrint.getWeight() != 1.0)
		{
			result.append(", {");
			result.append(rowToPrint.getWeight());
			result.append("}");
		}
		
		return result.toString();
    }

	@Override
	public Iterator<Vector> iterator() 
	{
		return new Iterator<Vector>()
			{
				private int nextIndex = 0;
				
				@Override
				public boolean hasNext()
				{
					return nextIndex < data.size();
				}

				@Override
				public Vector next()
				{
					Vector result = data.get(nextIndex);
					nextIndex++;
					return result;
				}
		
			};
	}
	
	public Stream<Vector> stream()
	{
		return StreamSupport.stream(spliterator(), false);
	}

	/**
	 * Determines if this matrix has any instance whose weight is not 1 (the default).
	 */
	public boolean hasInstanceWeightsNot1()
	{
		for (Vector v: data)
		{
			if (v.getWeight() != 1.0)
			{
				return true;
			}
		}
		return false;
	}
}
