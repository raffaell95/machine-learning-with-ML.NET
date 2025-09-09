using System;
using Microsoft.ML.Data;

namespace MachineLearning.Models;

public class CasaInputData
{   
    [LoadColumn(0)]
    public float Tamanho { get; set; }

    [LoadColumn(1)]
    public float Quartos { get; set; }

    [LoadColumn(2)]
    public float Preco { get; set; }
}
