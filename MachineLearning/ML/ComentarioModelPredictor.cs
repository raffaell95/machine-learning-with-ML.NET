using System;
using MachineLearning.Models;
using Microsoft.ML;

namespace MachineLearning.ML;

public class ComentarioModelPredictor
{
    private MLContext mLContext = new MLContext();
    private ITransformer modeloCarregado;

    public void CarregarModelo(string path)
    {
        DataViewSchema modeloSchema;
        modeloCarregado = mLContext.Model.Load(path, out modeloSchema);
    }

    public ComentarioPredictionResult Prever(ComentarioInputData novoComentario)
    {
        var predEngine = mLContext.Model.CreatePredictionEngine<ComentarioInputData, ComentarioPredictionResult>(
            modeloCarregado
        );

        return predEngine.Predict(novoComentario);
    }
}
