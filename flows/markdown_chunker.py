from metaflow import FlowSpec, step, current, schedule
from rag_tools.filetypes.markdown import Mixin as MarkdownMixin

@schedule(weekly=True)
class MarkdownChunker(FlowSpec, MarkdownMixin):

    @step
    def start(self):
        """
        Start the flow.
        Try to download the content from the repository.
        """

        # see config.py for the definition of repo_params
        # it is a list of dictionaries, 
        # that tell the Markdown tools where to look for content.
        # see /notebooks/markdonw_repo_explorer.ipynb for more details.
        from config.repo_params import SAMPLE_OSS_MARKDOWN_REPOS

        self.repo_params = SAMPLE_OSS_MARKDOWN_REPOS
        self.df = self.load_df_from_repo_list()
        self.next(self.end)

    @step
    def end(self):
        print("The flow has ended, with a dataframe of shape: {}".format(self.df.shape))
        print(
            f"""
            You can now use the dataframe to do whatever you want.
            To load it in a notebook, you can use the following code:

                from metaflow import Flow, namespace
                namespace('{current.namespace}')
                run = Run('{current.flow_name}/{current.run_id}')
                df = run.data.df
                print(df.shape)
        """)


if __name__ == "__main__":
    MarkdownChunker()