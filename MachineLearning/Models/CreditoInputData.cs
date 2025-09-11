using System;
using Microsoft.ML.Data;

namespace MachineLearning.Models;

public class CreditoInputData
{
    [LoadColumn(0)]
    public float RendaMensal { get; set; }

    [LoadColumn(1)]
    public float EstadoCivil { get; set; } // 0 - Salteiro, 1 - Casado, 2 - Divorciado, 3 - Viúvo, 4 - União Estável
    
    [LoadColumn(2)]
    public float NumeroDependentes { get; set; }

    [LoadColumn(3)]
    public float PossuiVeiculo { get; set; }

    [LoadColumn(4)]
    public float JaNegadoAntes { get; set; }

    [LoadColumn(5)]
    public bool Aprovado { get; set; }

}
