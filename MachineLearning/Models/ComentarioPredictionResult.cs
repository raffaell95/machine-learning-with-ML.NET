using System;
using Microsoft.ML.Data;

namespace MachineLearning.Models;

public class ComentarioPredictionResult
{
    [ColumnName("PredictedLabel")]
    public bool EhPositivo { get; set; }

    public float Probability { get; set; }
}
