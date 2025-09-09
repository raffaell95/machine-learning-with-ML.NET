using System;
using MachineLearning.Models;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace MachineLearning.ML;

public class CasaModelTrainer
{
    private MLContext mLContext = new MLContext();
    private IDataView dados;
    private ITransformer modeloTreinado;

    public void CarregarDadosCSV(string path)
    {
        dados = mLContext.Data.LoadFromTextFile<CasaInputData>(
            path: path,
            hasHeader: true,
            separatorChar: ','
        );
    }

    public void TreinarModelo()
    {
        var pipeline = mLContext.Transforms.Concatenate(
            "Features",
            nameof(CasaInputData.Tamanho),
            nameof(CasaInputData.Quartos)
        ).Append(mLContext.Regression.Trainers.LightGbm(
            labelColumnName: "Preco",
            numberOfIterations: 100
        ));
        /*.Append(mLContext.Regression.Trainers.Sdca(
            labelColumnName: "Preco",
            maximumNumberOfIterations: 100
        ));*/

        modeloTreinado = pipeline.Fit(dados);
    }

    public void SalvarModelo(string path)
    {
        mLContext.Model.Save(modeloTreinado, dados.Schema, path);
    }

    public void AvaliarModelo()
    {
        var previsoes = modeloTreinado.Transform(dados);

        var metricas = mLContext.Regression.Evaluate(
            previsoes,
            labelColumnName: "Preco",
            scoreColumnName: "Score"
        );

        Console.WriteLine($"MAE: {metricas.MeanAbsoluteError}");
        Console.WriteLine($"RMSE: {metricas.RootMeanSquaredError}");
        Console.WriteLine($"R2: {metricas.RSquared}");
    }

    public void AvaliarMelhorAlgoritino()
    {
        var experimentSettings = new RegressionExperimentSettings()
        {
            MaxExperimentTimeInSeconds = 60
        };

        var experiment = mLContext.Auto().CreateRegressionExperiment(experimentSettings);

        Console.WriteLine("Executando AutoML...");

        var result = experiment.Execute(dados, labelColumnName: "Preco");

        var melhorExecucao = result.BestRun;

        Console.WriteLine($"Melhor algoritimo: {melhorExecucao.TrainerName}");

        Console.WriteLine($"MAE: {melhorExecucao.ValidationMetrics.MeanAbsoluteError}");
        Console.WriteLine($"RMSE: {melhorExecucao.ValidationMetrics.RootMeanSquaredError}");
        Console.WriteLine($"R2: {melhorExecucao.ValidationMetrics.RSquared}");

        modeloTreinado = melhorExecucao.Model;
    }
}
