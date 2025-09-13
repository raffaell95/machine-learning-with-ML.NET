using System;
using Microsoft.ML.Data;

namespace MachineLearning.Models;

public class ComentarioInputData
{
    [LoadColumn(0)]
    public string Comentario { get; set; }

    [LoadColumn(1)]
    public bool EhPositivo { get; set; }
}
