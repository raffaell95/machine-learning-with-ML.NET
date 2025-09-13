using System;
using MachineLearning.Models;
using Microsoft.ML;

namespace MachineLearning.ML;

public class ComentarioModelTrainer
{
    private MLContext mLContext = new MLContext();
    private IDataView dados;
    private ITransformer modeloTreinado;

    public void CarregarDadosCSV(string path)
    {
        dados = mLContext.Data.LoadFromTextFile<ComentarioInputData>(
            path: path,
            hasHeader: true,
            separatorChar: '|'
        );
    }

    public void TreinarModelo()
    {
        // Criar pipeline
        var pipeline = mLContext.Transforms.Text.FeaturizeText(
            outputColumnName: "Features",
            inputColumnName: nameof(ComentarioInputData.Comentario)
        ).Append(mLContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: nameof(ComentarioInputData.EhPositivo),
            featureColumnName: "Features"
        ));

        modeloTreinado = pipeline.Fit(dados);
    }

    public void SalvarModelo(string path)
    {
        mLContext.Model.Save(modeloTreinado, dados.Schema, path);
    }
}
