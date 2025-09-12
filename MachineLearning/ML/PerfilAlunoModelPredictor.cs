using System;
using MachineLearning.Models;
using Microsoft.ML;

namespace MachineLearning.ML;

public class PerfilAlunoModelPredictor
{
    private MLContext mLContext = new MLContext();
    private ITransformer modeloCarregado;

    public void CarregarModelo(string path)
    {
        DataViewSchema modeloSchema;
        modeloCarregado = mLContext.Model.Load(path, out modeloSchema);
    }

    public PerfilAlunoPredictResult Prever(PerfilAlunoInputData novoAluno)
    {
        var predEngine = mLContext.Model.CreatePredictionEngine<PerfilAlunoInputData, PerfilAlunoPredictResult>(
            modeloCarregado
        );

        return predEngine.Predict(novoAluno);
    }
}
