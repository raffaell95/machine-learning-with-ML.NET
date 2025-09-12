using System;
using Microsoft.ML.Data;

namespace MachineLearning.Models;

public class ComprasCartaoPredictionResult
{
    [ColumnName("PredictedLabel")]
    public bool EhAnormal { get; set; }
    public float Score { get; set; }
}
