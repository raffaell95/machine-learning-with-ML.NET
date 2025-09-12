using System;
using Microsoft.ML.Data;

namespace MachineLearning.Models;

public class PerfilAlunoPredictResult
{
    [ColumnName("PredictedLabel")]
    public string PerfilPrevisto { get; set; }

    public float[] Score { get; set; }
    
}
