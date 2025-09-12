using System;
using System.Security.Cryptography.X509Certificates;
using MachineLearning.Models;
using Microsoft.ML;

namespace MachineLearning.ML;

public class ClienteModelTrainer
{
    private MLContext mLContext = new MLContext();
    private IDataView dados;
    private ITransformer modeloTreinado;

    public void CarregarDadosCSV(string path)
    {
        dados = mLContext.Data.LoadFromTextFile<ClienteInputData>(
            path: path,
            hasHeader: true,
            separatorChar: ','
        );
    }

    public void TreinarModelo()
    {
        // Criar pipeline
        var pipeline = mLContext.Transforms.Concatenate(
            "Features",
            nameof(ClienteInputData.CompraMes),
            nameof(ClienteInputData.ValorMedioGasto),
            nameof(ClienteInputData.VisitasSemana)
        ).Append(mLContext.Transforms.NormalizeMinMax("Features", "Features"))
        .Append(mLContext.Clustering.Trainers.KMeans(
            featureColumnName: "Features",
            numberOfClusters: 3
        ));

        // Treinar modelo
        modeloTreinado = pipeline.Fit(dados);
    }

    public void AnalisarGrupos()
    {
        var dadosComPredicao = modeloTreinado.Transform(dados);

        var resultados = mLContext.Data.CreateEnumerable<ClienteComGrupo>(
            dadosComPredicao,
            reuseRowObject: false
        );

        var grupos = resultados.GroupBy(x => x.GrupoPrevisto).Select(x => new
        {
            Grupo = x.Key,
            MediaComprasMes = x.Average(y => y.CompraMes),
            MediaValorGasto = x.Average(y => y.ValorMedioGasto),
            MediaVisitasPorSemana = x.Average(y => y.VisitasSemana),
            QuantidadeRegistros = x.Count()
        }).OrderBy(x => x.Grupo);

        foreach (var grupo in grupos)
        {
            Console.WriteLine($"Grupo {grupo.Grupo}");
            Console.WriteLine($"Compras/mês: {grupo.MediaComprasMes:F1}");
            Console.WriteLine($"Valor médio gasto: {grupo.MediaValorGasto:C}");
            Console.WriteLine($"Visitas/semana: {grupo.MediaVisitasPorSemana:F1}");
            Console.WriteLine($"Quantidade de registros no grupo: {grupo.QuantidadeRegistros}");
            Console.WriteLine();
        }
    }

    public void AvaliarModelo()
    {
        var previsoes = modeloTreinado.Transform(dados);

        var metricas = mLContext.Clustering.Evaluate(
            data: previsoes,
            scoreColumnName: "Score",
            featureColumnName: "Features"
        );

        Console.WriteLine($"Métricas da Distância: {metricas.AverageDistance}");
        Console.WriteLine($"Distância Máxima: {metricas.DaviesBouldinIndex}");
    }

    public void SalvarModelo(string path)
    {
        mLContext.Model.Save(modeloTreinado, dados.Schema, path);
    }
}
