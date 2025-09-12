using System;
using MachineLearning.Models;
using Microsoft.ML;

namespace MachineLearning.ML;

public class ComprasCartaoModelTrainer
{
    private MLContext mLContext = new MLContext();
    private IDataView dados;
    private ITransformer modeloTreinado;

    public void CarregarDadosCSV(string path)
    {
        dados = mLContext.Data.LoadFromTextFile<ComprasCartaoInputData>(
            path: path,
            hasHeader: true,
            separatorChar: ','
        );
    }

    public void TreinarModelo()
    {
        var pipeline = mLContext.Transforms.Concatenate(
            "Features",
            nameof(ComprasCartaoInputData.ValorCompra),
            nameof(ComprasCartaoInputData.Parcelado),
            nameof(ComprasCartaoInputData.HoraCompra)
        ).Append(mLContext.Transforms.NormalizeMinMax("Features", "Features"))
        .Append(mLContext.AnomalyDetection.Trainers.RandomizedPca(
            featureColumnName: "Features",
            rank: 2
        ));

        modeloTreinado = pipeline.Fit(dados);
    }

    public void AnalisarAnomalias()
    {
        var dadosComPredicao = modeloTreinado.Transform(dados);

        var resultados = mLContext.Data.CreateEnumerable<ComprasCartaoComAnomalia>(
            dadosComPredicao, reuseRowObject: false
        ).ToList();

        foreach (var resultado in resultados.Where(x => x.EhAnormal == true))
        {
            Console.WriteLine($"Compra: {resultado.ValorCompra} - Score: {resultado.Score}");
        }
    }

    public void SalvarModelo(string path)
    {
        mLContext.Model.Save(modeloTreinado, dados.Schema, path);
    }
}
