using System;
using MachineLearning.Models;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Trainers;

namespace MachineLearning.ML;

public class PerfilAlunoModelTrainer
{
    private MLContext mLContext = new MLContext();
    private IDataView dados;
    private ITransformer modeloTreinado;

    public void CarregarDadosCSV(string path)
    {
        dados = mLContext.Data.LoadFromTextFile<PerfilAlunoInputData>(
            path: path,
            hasHeader: true,
            separatorChar: ','
        );
    }

    public void TreinarModelo()
    {
        // Criar pipeline
        var pipeline = mLContext.Transforms.Conversion.MapValueToKey(
            "Label",
            nameof(PerfilAlunoInputData.PerfilAluno)
        ).Append(mLContext.Transforms.Concatenate(
            "Features",
            nameof(PerfilAlunoInputData.NotaProficienciaGramatical),
            nameof(PerfilAlunoInputData.NotaProficienciaGramatical),
            nameof(PerfilAlunoInputData.NotaConversacao)
        )).Append(mLContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        .Append(mLContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        // Treinar Modelo
        modeloTreinado = pipeline.Fit(dados);
    }

    public void AvaliarModelo()
    {
        var previsoes = modeloTreinado.Transform(dados);

        var metricas = mLContext.MulticlassClassification.Evaluate(
            data: previsoes,
            labelColumnName: "Label"
        );

        Console.WriteLine($"MicroAccuracy: {metricas.MicroAccuracy:P2}");
        Console.WriteLine($"MacroAccuracy: {metricas.MacroAccuracy:P2}");
        Console.WriteLine($"LogLoss: {metricas.LogLoss:P2}");
        Console.WriteLine($"LogLossReduction: {metricas.LogLossReduction:P2}");
    }

    public void AvaliarMelhorModelo()
    {
        var experimentSettings = new MulticlassExperimentSettings
        {
            MaxExperimentTimeInSeconds = 60
        };

        var experiment = mLContext.Auto().CreateMulticlassClassificationExperiment(
            experimentSettings
        );

        Console.WriteLine("Executando AutoML...");

        var result = experiment.Execute(dados, labelColumnName: nameof(PerfilAlunoInputData.PerfilAluno));

        var melhorExecucao = result.BestRun;

        Console.WriteLine($"Melhor algoritimo: {melhorExecucao.TrainerName}");
        Console.WriteLine($"MicroAccuracy: {melhorExecucao.ValidationMetrics.MicroAccuracy:P2}");
        Console.WriteLine($"MacroAccuracy: {melhorExecucao.ValidationMetrics.MacroAccuracy:P2}");
        Console.WriteLine($"LogLoss: {melhorExecucao.ValidationMetrics.LogLoss:P2}");
        Console.WriteLine($"LogLossReduction: {melhorExecucao.ValidationMetrics.LogLossReduction:P2}");

        modeloTreinado = melhorExecucao.Model;
    }

    public void SalvarModelo(string path)
    {
        mLContext.Model.Save(modeloTreinado, dados.Schema, path);
    }
}
