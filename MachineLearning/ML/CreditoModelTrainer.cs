using System;
using MachineLearning.Models;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace MachineLearning.ML;

public class CreditoModelTrainer
{
    private MLContext mLContext = new MLContext();
    private IDataView dados;
    private ITransformer modeloTreinado;

    public void CarregarDadosCSV(string path)
    {
        dados = mLContext.Data.LoadFromTextFile<CreditoInputData>(
            path: path,
            hasHeader: true,
            separatorChar: ','
        );
    }

    public void TreinarModelo()
    {
        var pipeline = mLContext.Transforms.Concatenate(
            "Features",
            nameof(CreditoInputData.RendaMensal),
            nameof(CreditoInputData.EstadoCivil),
            nameof(CreditoInputData.NumeroDependentes),
            nameof(CreditoInputData.PossuiVeiculo),
            nameof(CreditoInputData.JaNegadoAntes)
        ).Append(mLContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
            labelColumnName: "Aprovado"
        ));

        // Treinar Modelo
        modeloTreinado = pipeline.Fit(dados);
    }

    public void AvaliarModelo()
    {
        var previsoes = modeloTreinado.Transform(dados);

        var metricas = mLContext.BinaryClassification.Evaluate(
            data: previsoes,
            labelColumnName: nameof(CreditoInputData.Aprovado)
        );

        // Resultados da avaliação
        Console.WriteLine($"Acurácia: {metricas.Accuracy:P2}");
        Console.WriteLine($"Precisão: {metricas.PositivePrecision:P2}");
        Console.WriteLine($"Recall: {metricas.PositiveRecall:P2}");
        Console.WriteLine($"F1-Score: {metricas.F1Score:P2}");

    }

    public void AvaliarMelhorModelo()
    {
        var experimentSettings = new BinaryExperimentSettings
        {
            MaxExperimentTimeInSeconds = 60
        };

        var experiment = mLContext.Auto()
            .CreateBinaryClassificationExperiment(experimentSettings);

        Console.WriteLine("Executando AutoML...");

        var result = experiment.Execute(dados, labelColumnName: nameof(CreditoInputData.Aprovado));

        var melhorExecucao = result.BestRun;

        Console.WriteLine($"Melhor algoritimo: {melhorExecucao.TrainerName}");
        Console.WriteLine($"Melhor Acuracia: {melhorExecucao.ValidationMetrics.Accuracy}");
        Console.WriteLine($"Melhor Precisão: {melhorExecucao.ValidationMetrics.PositivePrecision}");
        Console.WriteLine($"Melhor Recall: {melhorExecucao.ValidationMetrics.PositiveRecall}");
        Console.WriteLine($"Melhor F1-Score: {melhorExecucao.ValidationMetrics.F1Score}");

        modeloTreinado = melhorExecucao.Model;
    }

    public void SalvarModelo(string path)
    {
        mLContext.Model.Save(modeloTreinado, dados.Schema, path);
    }
}
