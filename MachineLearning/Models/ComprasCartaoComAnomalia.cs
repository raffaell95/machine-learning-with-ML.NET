using System;
using Microsoft.ML.Data;

namespace MachineLearning.Models;

public class ComprasCartaoComAnomalia
{
    public float ValorCompra { get; set; }
    public float Parcelado { get; set; }
    public float HoraCompra { get; set; }

    [ColumnName("PredictedLabel")]
    public bool EhAnormal { get; set; }
    public float Score { get; set; }
}
