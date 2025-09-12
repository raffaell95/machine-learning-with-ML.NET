using System;
using Microsoft.ML.Data;

namespace MachineLearning.Models;

public class PerfilAlunoInputData
{
    [LoadColumn(0)]
    public float NotaProficienciaGramatical { get; set; }

    [LoadColumn(1)]
    public float CompreensaoOral { get; set; }

    [LoadColumn(2)]
    public float NotaConversacao { get; set; }

    [LoadColumn(3)]
    public string PerfilAluno { get; set; } 
}
