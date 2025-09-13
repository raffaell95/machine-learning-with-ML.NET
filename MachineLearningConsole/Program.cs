using MachineLearning.ML;
using MachineLearning.Models;


// ExemploRegressao();
// ExemploClassificacaoBinaria();
// ExemploClassificacaoMultiClasse();
// ExemploClustering();
// ExemploAnomalias();
// ExemploTexto();
ExemploRecomendacao();

void ExemploRecomendacao()
{
    var trainer = new RecomendacaoModelTrainer();
    trainer.CarregarDadosCSV(Path.Combine(AppContext.BaseDirectory, "recomendacoes.csv"));
    trainer.TreinarModelo();

    var pathModelo = Path.Combine(AppContext.BaseDirectory, "modelo_recomendacao.zip");
    trainer.SalvarModelo(pathModelo);

    var predictor = new RecomendacaoModelPredictor();
    predictor.CarregarModelo(pathModelo);

    var novaRecomendacao = new RecomendacaoInputData()
    {
        UsuarioId = 1,
        ProdutoId = 10
    };

    var resultado = predictor.Prever(novaRecomendacao);
    Console.WriteLine($"Score de recomendação: {resultado.Score}");
}

void ExemploTexto()
{
    var trainer = new ComentarioModelTrainer();
    trainer.CarregarDadosCSV(Path.Combine(AppContext.BaseDirectory, "comentarios.csv"));
    trainer.TreinarModelo();

    var pathModelo = Path.Combine(AppContext.BaseDirectory, "modelo_texto.zip");
    trainer.SalvarModelo(pathModelo);

    var predictor = new ComentarioModelPredictor();
    predictor.CarregarModelo(pathModelo);

    var exemplos = new[]
    {
        "Produto otimo, gostei do atendimento",
        "Demorou demais e chegou com defeito",
        "Excelente qualidade, muito satisfeito",
        "Não recomendo, tive uma experiencia ruim"
    };

    foreach (var texto in exemplos)
    {
        var resultado = predictor.Prever(new ComentarioInputData { Comentario = texto });
        Console.WriteLine($"Comentário: {texto}");
        Console.WriteLine($"Sentimento: {(resultado.EhPositivo ? "Positivo" : "Negativo")}");
        Console.WriteLine($"Probabilidade de ser positivo: {resultado.Probability:P1}");
    }
}

void ExemploAnomalias()
{
    var trainer = new ComprasCartaoModelTrainer();

    trainer.CarregarDadosCSV(Path.Combine(AppContext.BaseDirectory, "compras_cartao.csv"));
    trainer.TreinarModelo();
    trainer.AnalisarAnomalias();

    var pathModelo = Path.Combine(AppContext.BaseDirectory, "modelo_anomalias.zip");
    trainer.SalvarModelo(pathModelo);

    var predictor = new ComprasCartaoModelPredictor();
    predictor.CarregarModelo(pathModelo);

    var novaCompra = new ComprasCartaoInputData()
    {
        ValorCompra = 30000,
        Parcelado = 1,
        HoraCompra = 23
    };

    var resultado = predictor.Prever(novaCompra);
    Console.WriteLine($"Anomalia? {(resultado.EhAnormal ? "Sim" : "Não")}");
    Console.WriteLine($"Score: {resultado.Score:F4}");
}

void ExemploClustering()
{
    var trainer = new ClienteModelTrainer();

    trainer.CarregarDadosCSV(Path.Combine(AppContext.BaseDirectory, "clientes_agrupamento.csv"));
    trainer.TreinarModelo();
    trainer.AnalisarGrupos();
    trainer.AvaliarModelo();

    var pathModelo = Path.Combine(AppContext.BaseDirectory, "modelo_clustering.zip");
    trainer.SalvarModelo(pathModelo);

    var predictor = new ClienteModelPredictor();
    predictor.CarregarModelo(pathModelo);

    var novoCliente = new ClienteInputData()
    {
        CompraMes = 10,
        ValorMedioGasto = 1200,
        VisitasSemana = 3
    };

    var resultado = predictor.Prever(novoCliente);
    Console.WriteLine($"Novo cliente pertence ao grupo: {resultado.GrupoPrevisto}");
}

void ExemploClassificacaoMultiClasse()
{
    var trainer = new PerfilAlunoModelTrainer();

    trainer.CarregarDadosCSV(Path.Combine(AppContext.BaseDirectory, "perfil_aluno_idiomas.csv"));
    trainer.TreinarModelo();

    trainer.AvaliarModelo();
    trainer.AvaliarMelhorModelo();

    var pathModelo = Path.Combine(AppContext.BaseDirectory, "modelo_treinado_classificacao_multiclasse.zip");
    trainer.SalvarModelo(pathModelo);

    var predictor = new PerfilAlunoModelPredictor();
    predictor.CarregarModelo(pathModelo);

    var novoAluno = new PerfilAlunoInputData()
    {
        NotaProficienciaGramatical = 6.5f,
        CompreensaoOral = 7.0f,
        NotaConversacao = 5.5f
    };

    var resultado = predictor.Prever(novoAluno);

    Console.WriteLine($"Perfil previsto: {resultado.PerfilPrevisto}");

    Console.WriteLine("Pontuação por perfil:");

    var perfis = new[] { "Iniciante", "Intermediário", "Avançado" };

    for (int cont = 0; cont < resultado.Score.Length; cont++)
    {
        Console.WriteLine($"{perfis[cont]}: {resultado.Score[cont]:P2}");
    }
}

void ExemploClassificacaoBinaria()
{
    var trainer = new CreditoModelTrainer();

    trainer.CarregarDadosCSV(Path.Combine(AppContext.BaseDirectory, "aprovacao_credito.csv"));
    trainer.TreinarModelo();
    trainer.AvaliarModelo();
    trainer.AvaliarMelhorModelo();

    var pathModelo = Path.Combine(AppContext.BaseDirectory, "modelo_treinado_classificacao_binaria.zip");
    trainer.SalvarModelo(pathModelo);

    var predictor = new CreditoModelPrecictor();
    predictor.CarregarModelo(pathModelo);

    var novoCredito = new CreditoInputData()
    {
        RendaMensal = 4200f,
        EstadoCivil = 1,
        NumeroDependentes = 2,
        PossuiVeiculo = 1,
        JaNegadoAntes = 0
    };

    var resultado = predictor.Prever(novoCredito);
    Console.WriteLine($"Aprovado? {(resultado.PredicaoAprovado ? "Sim" : "Não")}");
    Console.WriteLine($"Probabilidade: {resultado.Probability:P2}");
}

void ExemploRegressao()
{
    var trainer = new CasaModelTrainer();

    trainer.CarregarDadosCSV(Path.Combine(AppContext.BaseDirectory, "casas_treinamento.csv"));
    trainer.TreinarModelo();

    trainer.AvaliarModelo();
    trainer.AvaliarMelhorAlgoritino();

    var pathModelo = Path.Combine(AppContext.BaseDirectory, "modelo_treinado_regressao.zip");
    trainer.SalvarModelo(pathModelo);

    var predictor = new CasaModelPredictor();
    predictor.CarregarModelo(pathModelo);

    var casaNova = new CasaInputData()
    {
        Tamanho = 85f,
        Quartos = 3
    };

    var resultado = predictor.Prever(casaNova);
    Console.WriteLine("O valor da casa nova é: " + resultado.PrecoPrevisto);
}