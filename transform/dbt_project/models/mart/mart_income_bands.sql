with cpi as (

    select
        date,
        state,
        division,
        cpi_index

    from {{ ref('stg_cpi_state')}}

),

hies as (

    select
        state,
        income_mean,
        income_median,
        expenditure_mean,
        gini,
        poverty_rate,
        income_band

    from {{ ref('stg_hies_state')}}

),

joined as (

    select
        c.date,
        c.state,
        c.division,
        c.cpi_index,
        h.income_mean,
        h.income_median,
        h.expenditure_mean,
        h.gini,
        h.poverty_rate,
        h.income_band

    from cpi c
    left join hies h on c.state = h.state

)

select * from joined