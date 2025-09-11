using System;
using MachineLearning.Models;
using Microsoft.ML;

namespace MachineLearning.ML;

public class CreditoModelPrecictor
{
    private MLContext mLContext = new MLContext();
    private ITransformer modeloCarregado;

    public void CarregarModelo(string path)
    {
        DataViewSchema modeloSchema;
        modeloCarregado = mLContext.Model.Load(path, out modeloSchema);
    }

    public CreditoPredictionResult Prever(CreditoInputData novoCredito)
    {
        var predEngine = mLContext.Model.CreatePredictionEngine<CreditoInputData, CreditoPredictionResult>(
            modeloCarregado
        );

        return predEngine.Predict(novoCredito);
    }
}
