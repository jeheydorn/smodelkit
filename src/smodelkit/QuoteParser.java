package smodelkit;

import java.util.Iterator;

/**
 * Parses a line of text using white space for a delimeter, with the exception
 * that any characters (including white space) inside single or double quotes
 * are returned in a single token. Also, single and double quotes can be escaped
 * by a backslash. 
 * @author joseph
 *
 */
public class QuoteParser implements Iterator<String>
{
	private String line;
	private int start;
	private boolean inQuotes;
	private boolean inDoubleQuoates;

	public QuoteParser(String line)
	{
		this.line = line.trim();
		start = 0;
	}
	
	@Override
	public boolean hasNext()
	{
		return start < line.length();
	}

	@Override
	public String next()
	{
		String nextToken;
		do
		{
			nextToken = findNextToken();
		}
		while (nextToken.length() == 0);
		return nextToken;
	}
	
	private String findNextToken()
	{
		int end = findNextNonQuotedWhiteSpace();
		if (start == end)
			throw new IllegalStateException("No next token.");
		//if (end == line.length() + 1)
		String result = line.substring(start, end).trim();
		start = end;
		return result;			
	}
	
	/**
	 * Returns the index + 1 of the next non-quoted white space character
	 * in line after start.
	 * @return
	 */
	private int findNextNonQuotedWhiteSpace()
	{
		int result = start;
		while(result < line.length())
		{
				// Ignore escaped quotes such as \'. 
			if (line.charAt(result) == '\'' && (result == 0 || line.charAt(result - 1) != '\\'))
				inQuotes = !inQuotes;
			if (line.charAt(result) == '"' && (result == 0 || line.charAt(result - 1) != '\\'))
				inDoubleQuoates = !inDoubleQuoates;
			
			if (!inQuotes && !inDoubleQuoates 
					&&  line.charAt(result) == (' ') || line.charAt(result) == '\t')
			{
				result++;
				break;
			}
			result++;
		}
		return result;
	}
}