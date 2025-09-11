using System;
using Microsoft.ML.Data;

namespace MachineLearning.Models;

public class CreditoPredictionResult
{
    [ColumnName("PredictedLabel")]
    public bool PredicaoAprovado { get; set; }

    [ColumnName("Probability")]
    public float Probability { get; set; }
}
