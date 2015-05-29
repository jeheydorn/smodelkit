package smodelkit.util;

import java.io.Serializable;

@SuppressWarnings("serial")
public class Bounds implements Serializable
{
	public double lower;
    public double upper;
    
    public Bounds()
    {
    	lower = Double.NEGATIVE_INFINITY;
    	upper = Double.POSITIVE_INFINITY;
    }
    
    public Bounds(double lower, double upper)
    {
    	this.lower = lower;
    	this.upper = upper;
    }
}
