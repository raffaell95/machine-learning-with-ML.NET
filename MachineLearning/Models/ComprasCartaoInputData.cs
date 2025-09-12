using System;
using Microsoft.ML.Data;

namespace MachineLearning.Models;

public class ComprasCartaoInputData
{
    [LoadColumn(0)]
    public float ValorCompra { get; set; }

    [LoadColumn(1)]
    public float Parcelado { get; set; }

    [LoadColumn(2)]
    public float HoraCompra { get; set; }
}
